import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {AgentSacConstructorProps, Transition} from '../@types';
import {NAME} from '../constants';
import {
  assertScalar,
  assertShape,
  createConvEncoder,
  getTrainableOnlyWeights,
  loadModelByName,
  saveModel,
} from '../utils';

import {AgentSac} from './AgentSac';

export class AgentSacTrainable extends AgentSac {

  actorOptimizer?: tf.Optimizer;
  _actionInput?: tf.SymbolicTensor;

  q1?: tf.LayersModel;
  q1Optimizer?: tf.Optimizer;
  q1Targ?: tf.LayersModel;

  q2?: tf.LayersModel;
  q2Optimizer?: tf.Optimizer;
  q2Targ?: tf.LayersModel;

  _logAlpha?: tf.Variable<tf.Rank.R0>;
  alphaOptimizer?: tf.Optimizer;

  constructor(props: Partial<AgentSacConstructorProps> = Object.create(null)) {
    super(props);
  }

  async initialize() {
    await super.initialize();
      
    this.actorOptimizer = tf.train.adam();

    this._actionInput = tf.input({batchShape: [null, this._nActions]});

    this.q1 = await this._getCritic(NAME.Q1);
    this.q1Optimizer = tf.train.adam();

    this.q2 = await this._getCritic(NAME.Q2);
    this.q2Optimizer = tf.train.adam();

    this.q1Targ = await this._getCritic(NAME.Q1_TARGET);
    this.q2Targ = await this._getCritic(NAME.Q2_TARGET);

    this._logAlpha = await this._getLogAlpha(NAME.ALPHA);
    this.alphaOptimizer = tf.train.adam();

    this.updateTargets(1);
  }

  async _getActor(name = 'actor'): Promise<tf.LayersModel> {
    const model = await super._getActor(name);
    model.trainable = true;
    return model;
  }

  async _getCritic(name = 'critic') {
    const checkpoint = await loadModelByName(name);
    if (checkpoint) return checkpoint

    const base = tf.layers.concatenate().apply([this._telemetryInput!, this._actionInput!])
    assert(base instanceof tf.SymbolicTensor);

    let outputs = tf.layers.dense({ units: 256, activation: 'relu' }).apply(
      this._sighted
        ? tf.layers.concatenate().apply([
            createConvEncoder(this._frameInputL!),
            createConvEncoder(this._frameInputR!),
            base,
          ])
        : base
    );

    outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs);
    outputs = tf.layers.dense({units: 1}).apply(outputs);

    assert(outputs instanceof tf.SymbolicTensor);

    const model = tf.model({
      inputs: this._sighted 
        ? [this._telemetryInput!, this._frameInputL!, this._frameInputR!, this._actionInput!] 
        : [this._telemetryInput!, this._actionInput!],
      outputs,
      name,
    })

    model.trainable = true;
    return model
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
      void this.updateTargets();
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
          const alpha = this._getAlpha()
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

          const alpha = this._getAlpha();
          const loss = alpha.mul(logPi).sub(criticValue);

          assertShape(freshAction, [this._batchSize, this._nActions]);
          assertShape(logPi, [this._batchSize, 1]);
          assertShape(q1Value, [this._batchSize, 1]);
          assertShape(criticValue, [this._batchSize, 1]);
          assertShape(loss, [this._batchSize, 1]);

          return tf.mean(loss)
      }
      
      const { value, grads } = tf.variableGrads(actorLossFunction, getTrainableOnlyWeights(this.actor!)) // true means trainableOnly
      
      this.actorOptimizer!.applyGradients(grads)
  }

  _trainAlpha(state: tf.Tensor[]) {
      const alphaLossFunction = (): tf.Scalar => {
          const sampledAction = this.sampleAction(state, true)
          assert(Array.isArray(sampledAction));

          const [, logPi] = sampledAction;

          const alpha = this._getAlpha()
          const loss = tf.scalar(-1).mul(
              alpha.mul( // TODO: not sure whether this should be alpha or logAlpha
                  logPi.add(tf.scalar(this._targetEntropy))
              )
          )

          assertShape(loss, [this._batchSize, 1]);
          return tf.mean(loss);
      }
      
      const { value, grads } = tf.variableGrads(alphaLossFunction, [this._logAlpha!]) // true means trainableOnly
      
      this.alphaOptimizer!.applyGradients(grads)
  }

  /**
   * Soft update target Q-networks.
   * 
   * @param {number} [tau = this._tau] - smoothing constant τ for exponentially moving average: `wTarg <- wTarg*(1-tau) + w*tau`
   */
  updateTargets(tau = this._tau) {
    const tau_t = tf.scalar(tau);
    const q1W = this.q1!.getWeights();
    const q2W = this.q2!.getWeights();
    const q1WTarg = this.q1Targ!.getWeights();
    const q2WTarg = this.q2Targ!.getWeights();
    const len = q1W.length;

    const calc = (w: tf.Tensor, wTarg: tf.Tensor) =>
      wTarg.mul(tf.scalar(1).sub(tau_t)).add(w.mul(tau_t));
      
    const w1 = [], w2 = []
    for (let i = 0; i < len; i++) {
      w1.push(calc(q1W[i], q1WTarg[i]));
      w2.push(calc(q2W[i], q2WTarg[i]));
    }
    this.q1Targ!.setWeights(w1);
    this.q2Targ!.setWeights(w2);
  }

  _getAlpha() {
    return tf.exp(this._logAlpha!);
  }

  async checkpoint() {
    await Promise.all([
      saveModel(this.actor!),
      saveModel(this.q1!),
      saveModel(this.q2!),
      saveModel(this.q1Targ!),
      saveModel(this.q2Targ!),
    ]);
  }

/**
         * Builds a log of entropy scale (α) for training.
         * 
         * @param {string} name 
         * @returns {tf.Variable} trainable variable for log entropy
         */
        async _getLogAlpha(name = 'alpha') {
            let logAlpha = 0.0

            const checkpoint = await loadModelByName(name);
            if (checkpoint) {
                const [weights] = checkpoint.getWeights();
                assert(weights);

                const arraySync = weights.arraySync();
                assert(Array.isArray(arraySync));

                const [children] = arraySync;
                assert(Array.isArray(children));

                const [child] = children;
                assert(typeof child === 'number');

                logAlpha = child;
            } else {
                const model = tf.sequential({ name });
                model.add(tf.layers.dense({ units: 1, inputShape: [1], useBias: false }))
                model.setWeights([tf.tensor([logAlpha], [1, 1])])
            }

            return tf.variable(tf.scalar(logAlpha), true) // true -> trainable
        }

}
