import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {AgentSacConstructorProps} from '../@types';
import {
  EPSILON,
  LOG_STD_MAX,
  LOG_STD_MIN,
  NAME,
} from '../constants';

import {Initializable} from './Initializable';
import {createConvEncoder, loadModelByName} from '../utils';

export class AgentSac extends Initializable {

  /* constructor */
  _batchSize: number;
  _frameShape: readonly number[];
  _nFrames: number;
  _nActions: number;
  _nTelemetry: number;
  _gamma: number;
  _tau: number;
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

  constructor({
    batchSize = 1, 
    frameShape = [25, 25, 3], 
    nFrames = 1, // Number of stacked frames per state
    nActions = 3, // 3 - impuls, 3 - RGB color
    nTelemetry = 10, // 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
    gamma = 0.99, // Discount factor (γ)
    tau = 5e-3, // Target smoothing coefficient (τ)
    sighted = true,
    rewardScale = 10
  }: Partial<AgentSacConstructorProps> = Object.create(null)) {
    super();
    this._batchSize = batchSize;
    this._frameShape = frameShape;
    this._nFrames = nFrames;
    this._nActions = nActions;
    this._nTelemetry = nTelemetry;
    this._gamma = gamma;
    this._tau = tau;
    this._sighted = sighted;
    this._rewardScale = rewardScale;
    this._frameStackShape = [...this._frameShape.slice(0, 2), this._frameShape[2] * this._nFrames] as [number, number, number];
    // https://github.com/rail-berkeley/softlearning/blob/13cf187cc93d90f7c217ea2845067491c3c65464/softlearning/algorithms/sac.py#L37
    this._targetEntropy = -nActions;
  }

  async initialize() {
    await super.initialize();

    this._frameInputL = tf.input({batchShape : [null, ...this._frameStackShape]});
    this._frameInputR = tf.input({batchShape : [null, ...this._frameStackShape]});
    this._telemetryInput = tf.input({batchShape : [null, this._nTelemetry]});
      
    this.actor = await this._getActor(NAME.ACTOR);
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
            const checkpoint = await loadModelByName(name);
            if (checkpoint) return checkpoint;

            let outputs = tf.layers.dense({ units: 256, activation: 'relu' }).apply(
              this._sighted
                ? tf.layers.concatenate().apply([
                  createConvEncoder(this._frameInputL!),
                  createConvEncoder(this._frameInputR!),
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
            return model;
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

        /**
         * Saves a model to the storage.
         * 
         * @param {tf.LayersModel} model 
         */
        

        
        
        
    }
