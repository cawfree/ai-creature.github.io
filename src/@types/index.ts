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
  // Whether the actor is trainable
  // for tests
  readonly sighted: boolean;
  readonly rewardScale: number;
};

export type AgentSacInstanceProps =
  & AgentSacConstructorProps
  & {
    readonly actor: tf.LayersModel; 
    readonly frameShape: readonly number[];
    readonly targetEntropy: number;
    readonly frameInputL: tf.SymbolicTensor;
    readonly frameInputR: tf.SymbolicTensor;
    readonly frameStackShape: [number, number, number];
    readonly telemetryInput: tf.SymbolicTensor;
  };

export type AgentSacTrainableInstanceProps =
  & AgentSacInstanceProps
  & {
    readonly actorOptimizer: tf.Optimizer;
    readonly actionInput: tf.SymbolicTensor;
    readonly q1: tf.LayersModel;
    readonly q1Optimizer: tf.Optimizer;
    readonly q1Targ: tf.LayersModel;
    readonly q2: tf.LayersModel;
    readonly q2Optimizer: tf.Optimizer;
    readonly q2Targ: tf.LayersModel;
    readonly logAlphaModel: tf.LayersModel;
    readonly logAlpha: tf.Variable<tf.Rank.R0>;
    readonly alphaOptimizer: tf.Optimizer;
    readonly tau: number;
  };

export type AgentSacSampleActionCallbackProps = {
  readonly state: tf.Tensor[];
};

export type AgentSacSampleActionCallback = (
  props: AgentSacSampleActionCallbackProps
) => [pi: tf.Tensor, logPi: tf.Tensor];

export type AgentSacTrainableCheckpointCallback = () => Promise<void>;

export type AgentSacInstance = {
  // TODO: Shouldn't expose this, use high-level writers.
  readonly actor: tf.LayersModel; 
  readonly batchSize: number;
  readonly frameShape: readonly number[];
  readonly frameStackShape: [number, number, number];
  readonly nActions: number;
  readonly nFrames: number;
  readonly nTelemetry: number;
  readonly sampleAction: AgentSacSampleActionCallback;
};

export type AgentSacTrainableTrainCallbackProps = {
  readonly transition: Omit<Transition, 'id' | 'priority'>;
};

export type AgentSacTrainableTrainCallback = (
  props: AgentSacTrainableTrainCallbackProps
) => void;

export type AgentSacTrainableInstance =
  & AgentSacInstance
  & {
    readonly checkpoint: AgentSacTrainableCheckpointCallback;
    readonly train: AgentSacTrainableTrainCallback;
  };

