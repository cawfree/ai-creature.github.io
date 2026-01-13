import * as tf from '@tensorflow/tfjs';

import {AgentSacInstanceProps} from '../@types';
import {sampleActionFrom} from '../utils';

import {Initializable} from './Initializable';

export class AgentSac extends Initializable {

  /* constructor */
  _batchSize: number;
  _frameShape: readonly number[];
  _nFrames: number;
  _nActions: number;
  _nTelemetry: number;
  _gamma: number;
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
    this._sighted = props.sighted;
    this._rewardScale = props.rewardScale;
    this._frameStackShape = props.frameStackShape;
    this._targetEntropy = props.targetEntropy;
    this._frameInputL = props.frameInputL;
    this._frameInputR = props.frameInputR;
    this._telemetryInput = props.telemetryInput;
    this.actor = props.actor;
  }

  sampleAction(state: tf.Tensor[], withLogProbs: boolean = false) {
    return sampleActionFrom({
      actor: this.actor!,
      batchSize: this._batchSize!,
      sighted: this._sighted,
      state,
      withLogProbs,
    });
  }
        
}
