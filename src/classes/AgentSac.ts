import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {AgentSacConstructorProps} from '../@types';
import {
  EPSILON,
  LOG_STD_MAX,
  LOG_STD_MIN,
  NAME,
  VERSION,
} from '../constants';

export class AgentSac {

  /* constructor */
  _batchSize: number;
  _frameShape: readonly number[];
  _nFrames: number;
  _nActions: number;
  _nTelemetry: number;
  _gamma: number;
  _tau: number;
  //_trainable: boolean;
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

  // TODO: idk.
  _logAlphaPlaceholder?: tf.LayersModel;

  constructor({
    batchSize = 1, 
    frameShape = [25, 25, 3], 
    nFrames = 1, // Number of stacked frames per state
    nActions = 3, // 3 - impuls, 3 - RGB color
    nTelemetry = 10, // 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
    gamma = 0.99, // Discount factor (γ)
    tau = 5e-3, // Target smoothing coefficient (τ)
    //trainable = true, // Whether the actor is trainable
    verbose = false,
    forced = false, // force to create fresh models (not from checkpoint)
    prefix = '', // for tests,
    sighted = true,
    rewardScale = 10
  }: Partial<AgentSacConstructorProps> = Object.create(null)) {
    this._batchSize = batchSize;
    this._frameShape = frameShape;
    this._nFrames = nFrames;
    this._nActions = nActions;
    this._nTelemetry = nTelemetry;
    this._gamma = gamma;
    this._tau = tau;
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
    this._inited = true;

    this._frameInputL = tf.input({batchShape : [null, ...this._frameStackShape]});
    this._frameInputR = tf.input({batchShape : [null, ...this._frameStackShape]});
    this._telemetryInput = tf.input({batchShape : [null, this._nTelemetry]});
      
    this.actor = await this._getActor(this._prefix + NAME.ACTOR);
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
        async _getActor(name = 'actor'): Promise<tf.LayersModel> {
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

            if (this._verbose) {
                console.log('==========================')
                console.log('==========================')
                console.log('Actor ' + name + ': ')

                model.summary()
            }

            return model;
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
