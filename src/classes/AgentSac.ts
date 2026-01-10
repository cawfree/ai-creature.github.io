import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {AgentSacProps, Transition} from '../@types';
import {
  assertScalar,
  assertShape,
  getTrainableOnlyWeights,
} from '../utils';

const VERSION = 84;

const LOG_STD_MIN = -20;
const LOG_STD_MAX = 2;
const EPSILON = 1e-8;
const NAME = {
  ACTOR: 'actor',
  Q1: 'q1',
  Q2: 'q2',        
  Q1_TARGET: 'q1-target',
  Q2_TARGET: 'q2-target',
  ALPHA: 'alpha'
};

export class AgentSac {

  /* constructor */
  _batchSize: number;
  _frameShape: readonly number[];
  _nFrames: number;
  _nActions: number;
  _nTelemetry: number;
  _gamma: number;
  _tau: number;
  _trainable: boolean;
  _verbose: boolean;
  _inited: boolean;
  _prefix: string;
  _forced: boolean;
  _sighted: boolean;
  _rewardScale: number;
  _frameStackShape: [number, number, number];
  _targetEntropy: number;

  /* initialization */
  _frameInputL?: tf.SymbolicTensor;
  _frameInputR?: tf.SymbolicTensor;
  _telemetryInput?: tf.SymbolicTensor;

  /* actor */
  actor?: tf.LayersModel;
  actorOptimizer?: tf.Optimizer;

  /* critic */
  q1?: tf.LayersModel;
  q1Targ?: tf.LayersModel;
  q1Optimizer?: tf.Optimizer;

  q2?: tf.LayersModel;
  q2Targ?: tf.LayersModel;
  q2Optimizer?: tf.Optimizer;

  // TODO: idk.
  _actionInput?: tf.SymbolicTensor;
  _logAlpha?: tf.Variable<tf.Rank.R0>;
  alphaOptimizer?: tf.Optimizer;
  _logAlphaPlaceholder?: tf.LayersModel;

  constructor({
    batchSize = 1, 
    frameShape = [25, 25, 3], 
    nFrames = 1, // Number of stacked frames per state
    nActions = 3, // 3 - impuls, 3 - RGB color
    nTelemetry = 10, // 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
    gamma = 0.99, // Discount factor (γ)
    tau = 5e-3, // Target smoothing coefficient (τ)
    trainable = true, // Whether the actor is trainable
    verbose = false,
    forced = false, // force to create fresh models (not from checkpoint)
    prefix = '', // for tests,
    sighted = true,
    rewardScale = 10
  }: Partial<AgentSacProps> = Object.create(null)) {
    this._batchSize = batchSize;
    this._frameShape = frameShape;
    this._nFrames = nFrames;
    this._nActions = nActions;
    this._nTelemetry = nTelemetry;
    this._gamma = gamma;
    this._tau = tau;
    this._trainable = trainable;
    this._verbose = verbose;
    this._inited = false;
    this._prefix = (prefix === '' ? '' : prefix + '-');
    this._forced = forced;
    this._sighted = sighted;
    this._rewardScale = rewardScale;
    this._frameStackShape = [...this._frameShape.slice(0, 2), this._frameShape[2] * this._nFrames] as [number, number, number];
    // https://github.com/rail-berkeley/softlearning/blob/13cf187cc93d90f7c217ea2845067491c3c65464/softlearning/algorithms/sac.py#L37
    this._targetEntropy = -nActions;
  }

  async init() {
    if (this._inited) throw Error('щ（ﾟДﾟщ）');

    this._frameInputL = tf.input({batchShape : [null, ...this._frameStackShape]});
    this._frameInputR = tf.input({batchShape : [null, ...this._frameStackShape]});
    this._telemetryInput = tf.input({batchShape : [null, this._nTelemetry]});
      
    this.actor = await this._getActor(this._prefix + NAME.ACTOR, this._trainable);
      
    if (!this._trainable) return;
      
    this.actorOptimizer = tf.train.adam();

    this._actionInput = tf.input({batchShape: [null, this._nActions]});

    this.q1 = await this._getCritic(this._prefix + NAME.Q1);
    this.q1Optimizer = tf.train.adam();

    this.q2 = await this._getCritic(this._prefix + NAME.Q2);
    this.q2Optimizer = tf.train.adam();

    // true for batch norm
    this.q1Targ = await this._getCritic(this._prefix + NAME.Q1_TARGET, true /* batch_normalization */);
    this.q2Targ = await this._getCritic(this._prefix + NAME.Q2_TARGET, true /* batch_normalization */);

    this._logAlpha = await this._getLogAlpha(this._prefix + NAME.ALPHA);
    this.alphaOptimizer = tf.train.adam();

    this.updateTargets(1);

    this._inited = true;
  }

  train({ state, action, reward, nextState }: Omit<Transition, 'id' | 'priority'>): void {
    if (!this._trainable) throw new Error('Actor is not trainable')
    
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

        /**
         * Train Q-networks.
         * 
         * @param {{ state, action, reward, nextState }} transition - transition
         */
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
                
                if (this._verbose) console.log(q!.name + ' Loss: ' + value.arraySync())
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

            if (this._verbose) console.log('Actor Loss: ' + value.arraySync())
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
            
            if (this._verbose) console.log('Alpha Loss: ' + value.arraySync(), tf.exp(this._logAlpha!).arraySync())
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

        /**
         * Returns actions sampled from normal distribution using means and stds predicted by the actor.
         * 
         * @param {Tensor[]} state - state
         * @param {Tensor} [withLogProbs = false] - whether return log probabilities
         * @returns {Tensor || Tensor[]} action and log policy
         */
        sampleAction(state: tf.Tensor[], withLogProbs: boolean = false) { // timer ~3ms
            return tf.tidy(() => {
                const prediction = this.actor!.predict(this._sighted ? state : state[0], {batchSize: this._batchSize})
                assert(Array.isArray(prediction));

                let [mu, logStd] = prediction;

                // https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/torch/sac/policies/gaussian_policy.py#L106
                logStd = tf.clipByValue(logStd, LOG_STD_MIN, LOG_STD_MAX) 
                
                const std = tf.exp(logStd)

                // sample normal N(mu = 0, std = 1)
                const normal = tf.randomNormal(mu.shape, 0, 1.0)
        
                // reparameterization trick: z = mu + std * epsilon
                let pi = mu.add(std.mul(normal))

                let logPi = this._gaussianLikelihood(pi, mu, logStd);

                ;({ pi, logPi } = this._applySquashing(pi, mu, logPi))

                if (!withLogProbs) return pi
        
                return [pi, logPi]
            })
        }

        /**
         * Calculates log probability of normal distribution https://en.wikipedia.org/wiki/Log_probability.
         * Converted to js from https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/distributions/normal.py#L183
         * 
         * @param {Tensor} x - sample from normal distribution with mean `mu` and std `std`
         * @param {Tensor} mu - mean
         * @param {Tensor} std - standart deviation
         * @returns {Tensor} log probability
         */
        _logProb(x: tf.Tensor, mu: tf.Tensor, std: tf.Tensor): tf.Tensor  {
            const logUnnormalized = tf.scalar(-0.5).mul(
                tf.squaredDifference(x.div(std), mu.div(std))
            )
            const logNormalization = tf.scalar(0.5 * Math.log(2 * Math.PI)).add(tf.log(std))
        
            return logUnnormalized.sub(logNormalization)
        }

        /**
         * Gaussian likelihood.
         * Translated from https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/core.py#L24
         * 
         * @param {Tensor} x - sample from normal distribution with mean `mu` and std `exp(logStd)`
         * @param {Tensor} mu - mean
         * @param {Tensor} logStd - log of standart deviation
         * @returns {Tensor} log probability
         */
        _gaussianLikelihood(x: tf.Tensor, mu: tf.Tensor, logStd: tf.Tensor): tf.Tensor {
            // pre_sum = -0.5 * (
            //     ((x-mu)/(tf.exp(log_std)+EPS))**2 
            //     + 2*log_std 
            //     + np.log(2*np.pi)
            // )

            const preSum = tf.scalar(-0.5).mul(
                x.sub(mu).div(
                    tf.exp(logStd).add(tf.scalar(EPSILON))
                ).square()
                .add(tf.scalar(2).mul(logStd))
                .add(tf.scalar(Math.log(2 * Math.PI)))
            )

            return tf.sum(preSum, 1, true)
        }

        /**
         * Adjustment to log probability when squashing action with tanh
         * Enforcing Action Bounds formula derivation https://stats.stackexchange.com/questions/239588/derivation-of-change-of-variables-of-a-probability-density-function
         * Translated from https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/core.py#L48
         * 
         * @param {*} pi - policy sample
         * @param {*} mu - mean
         * @param {*} logPi - log probability
         * @returns {{ pi, mu, logPi }} squashed and adjasted input
         */
        _applySquashing(pi: tf.Tensor, mu: tf.Tensor, logPi: tf.Tensor) {
            // logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

            const adj = tf.scalar(2).mul(
                tf.scalar(Math.log(2))
                .sub(pi)
                .sub(tf.softplus(
                    tf.scalar(-2).mul(pi)
                ))
            )

            logPi = logPi.sub(tf.sum(adj, 1, true))
            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

            return { pi, mu, logPi }
        }

        /**
         * Builds actor network model.
         * 
         * @param {string} [name = 'actor'] - name of the model
         * @param {string} trainable - whether a critic is trainable
         * @returns {tf.LayersModel} model
         */
        async _getActor(name = 'actor', trainable = true): Promise<tf.LayersModel> {
            const checkpoint = await this._loadCheckpoint(name)
            if (checkpoint) return checkpoint

            let outputs = tf.layers.dense({ units: 256, activation: 'relu' }).apply(
              this._sighted
                ? tf.layers.concatenate().apply([
                  this._getConvEncoder(this._frameInputL!),
                  this._getConvEncoder(this._frameInputR!),
                  this._telemetryInput!,
                ])
                : this._telemetryInput!
            );

            outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs)

            const mu     = tf.layers.dense({units: this._nActions}).apply(outputs)
            const logStd = tf.layers.dense({units: this._nActions}).apply(outputs)

            assert(mu instanceof tf.SymbolicTensor && logStd instanceof tf.SymbolicTensor);

            const model = tf.model({
                inputs:
                    this._sighted
                        ? [this._telemetryInput!, this._frameInputL!, this._frameInputR!]
                        : [this._telemetryInput!],
                outputs: [mu, logStd],
                name,
            })
            model.trainable = trainable

            if (this._verbose) {
                console.log('==========================')
                console.log('==========================')
                console.log('Actor ' + name + ': ')

                model.summary()
            }

            return model;
        }

        /**
         * Builds a critic network model.
         * 
         * @param {string} [name = 'critic'] - name of the model
         * @param {string} trainable - whether a critic is trainable
         * @returns {tf.LayersModel} model
         */
        async _getCritic(name = 'critic', trainable = true) {
            const checkpoint = await this._loadCheckpoint(name)
            if (checkpoint) return checkpoint

            const base = tf.layers.concatenate().apply([this._telemetryInput!, this._actionInput!])
            assert(base instanceof tf.SymbolicTensor);
            // outputs = tf.layers.dense({units: 128, activation: 'relu'}).apply(outputs)


            let outputs = tf.layers.dense({ units: 256, activation: 'relu' }).apply(
                this._sighted
                    ? tf.layers.concatenate().apply([
                        this._getConvEncoder(this._frameInputL!),
                        this._getConvEncoder(this._frameInputR!),
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

            model.trainable = trainable

            if (this._verbose) {
                console.log('==========================')
                console.log('==========================')
                console.log('CRITIC ' + name + ': ')
        
                model.summary()
            }

            return model
        }

        // _encoder = null
        // _getConvEncoder(inputs) {
        //     if (!this._encoder)
        //         this._encoder = this.__getConvEncoder(inputs)
            
        //     return this._encoder
        // }

        /**
         * Builds convolutional part of a network.
         * 
         * @param {Tensor} inputs - input for the conv layers
         * @returns outputs
         */
         _getConvEncoder(inputs: tf.SymbolicTensor): tf.SymbolicTensor {

            const kernelSize = 3
            const padding = 'valid'
            const poolSize = 3
            const strides = 1
            // const depthwiseInitializer = 'heNormal'
            // const pointwiseInitializer = 'heNormal'
            const kernelInitializer = 'glorotNormal'
            const biasInitializer = 'glorotNormal'

            // 32x8x4 -> 64x4x2 -> 64x3x1 -> 64x4x1
            let outputs = tf.layers.conv2d({
                filters: 16,
                kernelSize: 5,
                strides: 2,
                padding,
                kernelInitializer,
                biasInitializer,
                activation: 'relu',
                trainable: true
            }).apply(inputs)
            outputs = tf.layers.maxPooling2d({poolSize:2}).apply(outputs)
            // 
            // outputs = tf.layers.layerNormalization().apply(outputs)

            outputs = tf.layers.conv2d({
                filters: 16,
                kernelSize: 3,
                strides: 1,
                padding,
                kernelInitializer,
                biasInitializer,
                activation: 'relu',
                trainable: true
            }).apply(outputs)
            outputs = tf.layers.maxPooling2d({poolSize:2}).apply(outputs)

            // outputs = tf.layers.layerNormalization().apply(outputs)
            
            // outputs = tf.layers.conv2d({
            //     filters: 12,
            //     kernelSize: 3,
            //     strides: 1,
            //     padding,
            //     kernelInitializer,
            //     biasInitializer,
            //     activation: 'relu',
            //     trainable: true
            // }).apply(outputs)

            // outputs = tf.layers.conv2d({
            //     filters: 10,
            //     kernelSize: 2,
            //     strides: 1,
            //     padding,
            //     kernelInitializer,
            //     biasInitializer,
            //     activation: 'relu',
            //     trainable: true
            // }).apply(outputs)

            // outputs = tf.layers.conv2d({
            //     filters: 64,
            //     kernelSize: 4,
            //     strides: 1,
            //     padding,
            //     kernelInitializer,
            //     biasInitializer,
            //     activation: 'relu'
            // }).apply(outputs)

            // outputs = tf.layers.batchNormalization().apply(outputs)

            // outputs = tf.layers.layerNormalization().apply(outputs)

            outputs = tf.layers.flatten().apply(outputs)

            // convOutputs = tf.layers.dense({units: 96, activation: 'relu'}).apply(convOutputs)

             console.log('here with the shit');
            assert(outputs instanceof tf.SymbolicTensor);
            return outputs
        }

        /**
         * Returns clipped alpha.
         * 
         * @returns {Tensor} entropy
         */
        _getAlpha() {
            // return tf.maximum(tf.exp(this._logAlpha), tf.scalar(this._minAlpha))
            return tf.exp(this._logAlpha!)
        }

        /**
         * Builds a log of entropy scale (α) for training.
         * 
         * @param {string} name 
         * @returns {tf.Variable} trainable variable for log entropy
         */
        async _getLogAlpha(name = 'alpha') {
            let logAlpha = 0.0

            const checkpoint = await this._loadCheckpoint(name)
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

                if (this._verbose)
                    console.log('Checkpoint alpha: ', logAlpha)
                    
                this._logAlphaPlaceholder = checkpoint
            } else {
                const model = tf.sequential({ name });
                model.add(tf.layers.dense({ units: 1, inputShape: [1], useBias: false }))
                model.setWeights([tf.tensor([logAlpha], [1, 1])])

                this._logAlphaPlaceholder = model
            }

            return tf.variable(tf.scalar(logAlpha), true) // true -> trainable
        }

        /**
         * Saves all agent's models to the storage.
         */
        async checkpoint() {
            if (!this._trainable) throw new Error('(╭ರ_ ⊙ )')

            this._logAlphaPlaceholder!.setWeights([tf.tensor([this._logAlpha!.arraySync()], [1, 1])])

            await Promise.all([
                this._saveCheckpoint(this.actor!),
                this._saveCheckpoint(this.q1!),
                this._saveCheckpoint(this.q2!),
                this._saveCheckpoint(this.q1Targ!),
                this._saveCheckpoint(this.q2Targ!),
                this._saveCheckpoint(this._logAlphaPlaceholder!)
            ])

            if (this._verbose) 
                console.log('Checkpoint succesfully saved')
        }

        /**
         * Saves a model to the storage.
         * 
         * @param {tf.LayersModel} model 
         */
        async _saveCheckpoint(model: tf.LayersModel) {
            const key = this._getChKey(model.name)
            const saveResults = await model.save(key)

            if (this._verbose) 
                console.log('Checkpoint saveResults', model.name, saveResults)
        }

        /**
         * Loads saved checkpoint from the storage.
         * 
         * @param {string} name model name
         * @returns {tf.LayersModel} model
         */
        async _loadCheckpoint(name: string) {
// return
            if (this._forced) {
                console.log('Forced to not load from the checkpoint ' + name)
                return
            }

            const key = this._getChKey(name)
            const modelsInfo = await tf.io.listModels()

            if (key in modelsInfo) {
                const model = await tf.loadLayersModel(key)

                if (this._verbose) 
                    console.log('Loaded checkpoint for ' + name)

                return model
            }
            
            if (this._verbose) 
                console.log('Checkpoint not found for ' + name)
        }
        
        /**
         * Builds the key for the model weights in LocalStorage.
         * 
         * @param {tf.LayersModel} name model name
         * @returns {string} key
         */
        _getChKey(name: string) {
            return 'indexeddb://' + name + '-' + VERSION
        }
    }
