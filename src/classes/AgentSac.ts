import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {AgentSacInstanceProps} from '../@types';
import {
  EPSILON,
  LOG_STD_MAX,
  LOG_STD_MIN,
} from '../constants';

import {Initializable} from './Initializable';
import {createActor, loadModelByName} from '../utils';

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
  _frameInputL: tf.SymbolicTensor;
  _frameInputR: tf.SymbolicTensor;
  _telemetryInput: tf.SymbolicTensor;

  /* actor */
  actor: tf.LayersModel; 

  constructor(props: AgentSacInstanceProps) {
    super();
    this._batchSize = props.batchSize;
    this._frameShape = props.frameShape;
    this._nFrames = props.nFrames;
    this._nActions = props.nActions;
    this._nTelemetry = props.nTelemetry;
    this._gamma = props.gamma;
    this._tau = props.tau;
    this._sighted = props.sighted;
    this._rewardScale = props.rewardScale;
    this._frameStackShape = props.frameStackShape;
    this._targetEntropy = props.targetEntropy;
    this._frameInputL = props.frameInputL;
    this._frameInputR = props.frameInputR;
    this._telemetryInput = props.telemetryInput;
    this.actor = props.actor;
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
        
    }
