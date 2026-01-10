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

export type AgentSacProps = {
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
  readonly trainable: boolean;
  readonly verbose: boolean;
  // force to create fresh models (not from checkpoint)
  readonly forced: boolean;
  // for tests
  readonly prefix: string;
  readonly sighted: boolean;
  readonly rewardScale: number;
};
