import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {AgentSacTrainableInstanceProps, Transition} from '../@types';
import {
  assertScalar,
  assertShape,
  getTrainableOnlyWeights,
  trainAlpha,
  updateTrainableTargets,
} from '../utils';

import {AgentSac} from './AgentSac';

export class AgentSacTrainable extends AgentSac {

  actorOptimizer: tf.Optimizer;
  _actionInput: tf.SymbolicTensor;

  q1: tf.LayersModel;
  q1Optimizer: tf.Optimizer;
  q1Targ: tf.LayersModel;

  q2: tf.LayersModel;
  q2Optimizer: tf.Optimizer;
  q2Targ: tf.LayersModel;

  alphaOptimizer: tf.Optimizer;

  logAlphaModel: tf.LayersModel;
  logAlpha: tf.Variable<tf.Rank.R0>;

  _tau: number;

  constructor(props: AgentSacTrainableInstanceProps) {
    super(props);
    this.actorOptimizer = props.actorOptimizer;
    this._actionInput = props.actionInput;
    this.q1 = props.q1;
    this.q1Optimizer = props.q1Optimizer;
    this.q1Targ = props.q1Targ;
    this.q2 = props.q2;
    this.q2Optimizer = props.q2Optimizer;
    this.q2Targ = props.q2Targ;
    this.alphaOptimizer = props.alphaOptimizer;
    this.logAlphaModel = props.logAlphaModel;
    this.logAlpha = props.logAlpha;
    this._tau = props.tau;
  }

  train({
    state,
    action,
    reward,
    nextState,
  }: Omit<Transition, 'id' | 'priority'>): void {
    return tf.tidy(() => {
      assertShape(state[0], [this._batchSize, this._nTelemetry]);
      assertShape(state[1], [this._batchSize, ...this._frameStackShape]);
      assertShape(action, [this._batchSize, this._nActions]);
      assertShape(reward, [this._batchSize, 1]);
      assertShape(nextState[0], [this._batchSize, this._nTelemetry]);
      assertShape(nextState[1], [this._batchSize, ...this._frameStackShape]);
    
      void this._trainCritics({state, action, reward, nextState});
      void this._trainActor(state);
      void this._trainAlpha(state);

      void updateTrainableTargets({
        q1: this.q1!,
        q1Targ: this.q1Targ!,
        q2: this.q2!,
        q2Targ: this.q2Targ!,
        tau: this._tau,
      });
    });
  }

  _trainCritics({ state, action, reward, nextState }: Omit<Transition, 'id' | 'priority'>) {
    const getQLossFunction = (() => {
          const sampledAction = this.sampleAction(nextState, true)
          assert(Array.isArray(sampledAction));

          const [nextFreshAction, logPi] = sampledAction;

          const q1TargValue = this.q1Targ!.predict(
              this._sighted ? [...nextState, nextFreshAction] : [nextState[0], nextFreshAction], 
              {batchSize: this._batchSize})
          const q2TargValue = this.q2Targ!.predict(
              this._sighted ? [...nextState, nextFreshAction] : [nextState[0], nextFreshAction], 
              {batchSize: this._batchSize})
          
          assert(!Array.isArray(q1TargValue) && !Array.isArray(q2TargValue));
          const qTargValue = tf.minimum(q1TargValue, q2TargValue)
  
          // y = r + γ*(1 - d)*(min(Q1Targ(s', a'), Q2Targ(s', a')) - α*log(π(s'))
      const alpha = tf.exp(this.logAlpha);
          const target = reward.mul(tf.scalar(this._rewardScale)).add(
              tf.scalar(this._gamma).mul(
                  qTargValue.sub(alpha.mul(logPi))
              )
          )
                      
          assertShape(nextFreshAction, [this._batchSize, this._nActions]);
          assertShape(logPi, [this._batchSize, 1]);
          assertShape(qTargValue, [this._batchSize, 1]);
          assertShape(target, [this._batchSize, 1]);
  
          return (q: tf.LayersModel) => (): tf.Scalar => {
              const qValue = q.predict(
                  this._sighted ? [...state, action] : [state[0], action],
                  {batchSize: this._batchSize})
              
              assert(!Array.isArray(qValue));
              
              const loss = tf.scalar(0.5).mul(tf.mean(qValue.sub(target).square()))
              assertShape(qValue, [this._batchSize, 1]);

              return assertScalar(loss);
          }
      })()
  
      for (const [q, optimizer] of [
          [this.q1, this.q1Optimizer] as const,
          [this.q2, this.q2Optimizer] as const
      ]) {
          const qLossFunction = getQLossFunction(q!)

          const { value, grads } = tf.variableGrads(qLossFunction, getTrainableOnlyWeights(q!));
          
          optimizer!.applyGradients(grads)
      }
  }

  /**
   * Train actor networks.
   * 
   * @param {state} state 
   */
  _trainActor(state: tf.Tensor[]) {
      // TODO: consider delayed update of policy and targets (if possible)
      const actorLossFunction = (): tf.Scalar => {
          const sampledAction = this.sampleAction(state, true)
          assert(Array.isArray(sampledAction));

          const [freshAction, logPi] = sampledAction;
          
          const q1Value = this.q1!.predict(
              this._sighted ? [...state, freshAction] : [state[0], freshAction],
              {batchSize: this._batchSize});
          const q2Value = this.q2!.predict(
              this._sighted ? [...state, freshAction] : [state[0], freshAction], 
              {batchSize: this._batchSize});
          
          assert(!Array.isArray(q1Value) && !Array.isArray(q2Value));
          const criticValue = tf.minimum(q1Value, q2Value);

          const alpha = tf.exp(this.logAlpha);
          const loss = alpha.mul(logPi).sub(criticValue);

          assertShape(freshAction, [this._batchSize, this._nActions]);
          assertShape(logPi, [this._batchSize, 1]);
          assertShape(q1Value, [this._batchSize, 1]);
          assertShape(criticValue, [this._batchSize, 1]);
          assertShape(loss, [this._batchSize, 1]);

          return tf.mean(loss)
      }
      
      const { grads } = tf.variableGrads(actorLossFunction, getTrainableOnlyWeights(this.actor!)) // true means trainableOnly
      
      this.actorOptimizer!.applyGradients(grads)
  }

  _trainAlpha(state: tf.Tensor[]) {
    return trainAlpha({
      actor: this.actor!,
      alphaOptimizer: this.alphaOptimizer!,
      batchSize: this._batchSize,
      logAlpha: this.logAlpha,
      sighted: this._sighted,
      state,
      targetEntropy: this._targetEntropy,
    });
  }

  //async checkpoint() {
  //  return saveCheckpoints({
  //    actor: this.actor!,
  //    q1: this.q1!,
  //    q1Targ: this.q1Targ!,
  //    q2: this.q2!,
  //    q2Targ: this.q2Targ!,
  //    logAlpha: this.logAlpha,
  //    logAlphaModel: this.logAlphaModel,
  //  });
  //}

}
