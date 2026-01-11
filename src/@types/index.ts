import * as tf from '@tensorflow/tfjs';

export type Transition = {
  readonly id: number;
  // TODO: maybe declare separately?
  readonly priority?: number;
  readonly state: tf.Tensor[];
  readonly nextState: tf.Tensor[];
  readonly action: tf.Tensor;
  readonly reward: tf.Tensor;
};

export type AgentSacConstructorProps = {
  readonly batchSize: number;
  readonly frameShape: readonly number[];
  // Number of stacked frames per state
  readonly nFrames: number;
  // 3 - impuls, 3 - RGB color
  readonly nActions: number;
  // 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
  readonly nTelemetry: number;
  // Discount factor (γ)
  readonly gamma: number;
  // Target smoothing coefficient (τ)
  readonly tau: number;
  // Whether the actor is trainable
  // for tests
  readonly sighted: boolean;
  readonly rewardScale: number;
};

export type AgentSacInstanceProps =
  & AgentSacConstructorProps
  & {
    readonly frameStackShape: [number, number, number];
    readonly targetEntropy: number;
    readonly frameInputL: tf.SymbolicTensor;
    readonly frameInputR: tf.SymbolicTensor;
    readonly telemetryInput: tf.SymbolicTensor;
    readonly actor: tf.LayersModel; 
  };
