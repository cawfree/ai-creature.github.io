import * as tf from '@tensorflow/tfjs';

export type TensorLike =
  | tf.Tensor<tf.Rank>
  | tf.Tensor<tf.Rank>[]
  | tf.SymbolicTensor
  | tf.SymbolicTensor[];

export type Transition<State> = {
  readonly id: number;
  // TODO: maybe declare separately?
  readonly priority?: number;
  readonly state: State;
  readonly nextState: State;
  readonly action: tf.Tensor;
  readonly reward: tf.Tensor;
};

// TODO: this is stacked state
export type VectorizedTransitions = {
  readonly state: tf.Tensor[];
  readonly nextState: tf.Tensor[];
  readonly action: tf.Tensor;
  readonly reward: tf.Tensor;
};

export type AgentSacConstructorProps = {
  readonly batchSize: number;
  // 3 - impuls, 3 - RGB color
  readonly nActions: number;
  // Discount factor (γ)
  readonly gamma: number;
  readonly rewardScale: number;
};

export type AgentSacInstanceProps<TensorsIn extends SymbolicTensors> =
  & AgentSacConstructorProps
  & {
    readonly actor: tf.LayersModel; 
    readonly tensorsIn: TensorsIn;
    readonly targetEntropy: number;
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

export type SampledAction = [pi: tf.Tensor, logPi: tf.Tensor];

export type AgentSacSampleActionCallback = (props: AgentSacSampleActionCallbackProps) => SampledAction;

export type AgentSacTrainableCheckpointCallback = () => Promise<void>;

export type SymbolicTensors = {
  readonly [key: string]: tf.SymbolicTensor,
};

export type AgentSacInstance = {
  // TODO: Shouldn't expose this, use high-level writers.
  readonly actor: tf.LayersModel; 
  readonly batchSize: number;
  readonly nActions: number;
  readonly sampleAction: AgentSacSampleActionCallback;
};

export type AgentSacTrainableTrainCallbackProps<State> = {
  readonly transitions: readonly Omit<Transition<State>, 'id' | 'priority'>[];
};

export type AgentSacTrainableTrainCallback<State> = (
  props: AgentSacTrainableTrainCallbackProps<State>
) => void;

export type AgentSacTrainableInstance<State> =
  & AgentSacInstance
  & {
    readonly checkpoint: AgentSacTrainableCheckpointCallback;
    readonly train: AgentSacTrainableTrainCallback<State>;
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

export type AgentSacGetActorCreateTensorsInCallbackProps = {};

export type AgentSacGetActorCreateTensorsInCallback<
  TensorsIn extends SymbolicTensors,
> = (props: AgentSacGetActorCreateTensorsInCallbackProps) => TensorsIn;

export type VectorizeTransitionsCallbackProps<State> = {
  readonly transitions: readonly Omit<Transition<State>, 'id' | 'priority'>[];
};

export type VectorizeTransitionsCallback<State> =
  (props: VectorizeTransitionsCallbackProps<State>) => VectorizedTransitions;
