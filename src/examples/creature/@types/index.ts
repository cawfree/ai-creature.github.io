import * as tf from '@tensorflow/tfjs';

export type CreatureTensorsIn = {
  readonly frameInputL: tf.SymbolicTensor;
  readonly frameInputR: tf.SymbolicTensor;
  readonly telemetryInput: tf.SymbolicTensor;
};