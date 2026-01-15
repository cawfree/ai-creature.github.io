import * as tf from '@tensorflow/tfjs';

export type TensorLike = tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] | tf.SymbolicTensor | tf.SymbolicTensor[];

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
  // 3 - impuls, 3 - RGB color
  readonly nActions: number;
  // 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
  readonly nTelemetry: number;
  // Discount factor (γ)
  readonly gamma: number;
  readonly rewardScale: number;
};

export type AgentSacInstanceProps<TensorsIn extends SymbolicTensors> =
  & AgentSacConstructorProps
  & {
    readonly actor: tf.LayersModel; 
    readonly frameShape: readonly number[];
    readonly tensorsIn: TensorsIn;
    readonly targetEntropy: number;
    readonly frameStackShape: [number, number, number];
  };

export type AgentSacTrainableInstanceProps<TensorsIn extends SymbolicTensors> =
  & AgentSacInstanceProps<TensorsIn>
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

export type SymbolicTensors = {
  readonly [key: string]: tf.SymbolicTensor,
};

export type AgentSacInstance = {
  // TODO: Shouldn't expose this, use high-level writers.
  readonly actor: tf.LayersModel; 
  readonly batchSize: number;
  readonly frameShape: readonly number[];
  readonly frameStackShape: [number, number, number];
  readonly nActions: number;
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

export type AgentSacGetPredictionArgsCallbackProps = {
  readonly state: tf.Tensor[];
};

export type AgentSacGetPredictionArgsCallback = (
  props: AgentSacGetPredictionArgsCallbackProps
) => tf.Tensor[] | tf.Tensor;

export type AgentSacGetActorInputTensorsCallbackProps<
  TensorsIn extends SymbolicTensors
> = {
  readonly tensorsIn: TensorsIn;
};

export type AgentSacGetActorInputTensorsCallback<
  TensorsIn extends SymbolicTensors,
> = (
  props: AgentSacGetActorInputTensorsCallbackProps<TensorsIn>
) => tf.SymbolicTensor[];

export type AgentSacGetActorExtractModelInputsCallbackProps<
  TensorsIn extends SymbolicTensors
> = {
  readonly tensorsIn: TensorsIn;
};

export type AgentSacGetActorExtractModelInputsCallback<
  TensorsIn extends SymbolicTensors,
> = (
  props: AgentSacGetActorExtractModelInputsCallbackProps<TensorsIn>
) => tf.SymbolicTensor[];

export type AgentSacGetActorCreateTensorsInCallbackProps = {
  // TODO: shouldn't be here
  readonly frameStackShape: [number, number, number];
  readonly nTelemetry: number;
};

export type AgentSacGetActorCreateTensorsInCallback<
  TensorsIn extends SymbolicTensors,
> = (props: AgentSacGetActorCreateTensorsInCallbackProps) => TensorsIn;

