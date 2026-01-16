import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';
import isEqual from 'react-fast-compare';

import {
  AgentSacConstructorProps,
  AgentSacGetActorCreateTensorsInCallback,
  AgentSacGetActorExtractModelInputsCallback,
  AgentSacGetActorInputTensorsCallback,
  AgentSacGetPredictionArgsCallback,
  AgentSacInstance,
  AgentSacInstanceProps,
  AgentSacSampleActionCallback,
  AgentSacSampleActionCallbackProps,
  AgentSacTrainableCheckpointCallback,
  AgentSacTrainableInstance,
  AgentSacTrainableInstanceProps,
  AgentSacTrainableTrainCallback,
  AgentSacTrainableTrainCallbackProps,
  SymbolicTensors,
  Transition,
  VectorizedTransitions,
} from '../@types';
import {EPSILON, LOG_STD_MAX, LOG_STD_MIN, VERSION} from '../constants';

// https://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
export const getRandomInt = (min: number, max: number)  => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const assertNumericArray = (e: unknown): number[] => {
  assert(Array.isArray(e));

  return e.map(i => {
    assert(typeof i === 'number');
    return i;
  });
}

export const assertShape = (tensor: tf.Tensor, shape: readonly number[]) =>
  assert(isEqual(tensor.shape, shape));

export const assertScalar = (t: tf.Tensor): tf.Scalar => {
  assert(t.rank === 0);
  return t as tf.Scalar;
};

const maybeConcatenateTensors = (t: tf.SymbolicTensor[]) => {
  assert(Array.isArray(t));
  return t.length > 1 ? tf.layers.concatenate().apply(t) : t;
};

const getTrainableOnlyWeights = (layersModel: tf.LayersModel) =>
  layersModel
    .getWeights(true /* trainableOnly */)
    .map(w => {
      assert(w instanceof tf.Variable);
      return w;
    });

const getModelKey = (modelName: string) => {
  assert(typeof modelName === 'string' && modelName.length);
  return `indexeddb://${modelName}-${VERSION}`;
};

export const saveModel = (
  model: tf.LayersModel
) => model.save(getModelKey(model.name));

export const loadModelByName = async (
  modelName: string
): Promise<tf.LayersModel | null> => {
  const key = getModelKey(modelName);
  const modelsInfo = await tf.io.listModels();
  if (key in modelsInfo) return tf.loadLayersModel(key);
  return null;
};

// https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/core.py#L24
export const gaussianLikelihood = (x: tf.Tensor, mu: tf.Tensor, logStd: tf.Tensor): tf.Tensor =>
  tf.sum(tf.scalar(-0.5).mul(x.sub(mu).div(tf.exp(logStd).add(tf.scalar(EPSILON))).square().add(tf.scalar(2).mul(logStd)).add(tf.scalar(Math.log(2 * Math.PI)))), 1, true);

// https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/tf1/sac/core.py#L48
export const applySquashing = (pi: tf.Tensor, mu: tf.Tensor, logPi: tf.Tensor) => {
  const adj = tf.scalar(2).mul(tf.scalar(Math.log(2)).sub(pi) .sub(tf.softplus(tf.scalar(-2).mul(pi))));
  logPi = logPi.sub(tf.sum(adj, 1, true));
  mu = tf.tanh(mu);
  pi = tf.tanh(pi);
  return {pi, mu, logPi};
};

export const sampleActionFrom = ({
  actor,
  batchSize,
  getPredictionArgs,
  state,
}: {
  readonly actor: tf.LayersModel;
  readonly batchSize: number;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
  readonly state: tf.Tensor[];
}): [pi: tf.Tensor, logPi: tf.Tensor] => tf.tidy(() => {
  const prediction = actor.predict(getPredictionArgs({state}), {batchSize});
  assert(Array.isArray(prediction));

  let [mu, logStd] = prediction;

  // https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/torch/sac/policies/gaussian_policy.py#L106
  logStd = tf.clipByValue(logStd, LOG_STD_MIN, LOG_STD_MAX);
  
  const std = tf.exp(logStd);

  // sample normal N(mu = 0, std = 1)
  const normal = tf.randomNormal(mu.shape, 0, 1.0);

  // reparameterization trick: z = mu + std * epsilon
  let pi = mu.add(std.mul(normal));
  let logPi = gaussianLikelihood(pi, mu, logStd);

  ({pi, logPi} = applySquashing(pi, mu, logPi));
  return [pi, logPi];
});

export const createActor = async <
  TensorsIn extends SymbolicTensors
>({
  tensorsIn,
  getActorExtractModelInputs,
  getActorInputTensors,
  nActions,
  name,
}: {
  readonly tensorsIn: TensorsIn;
  readonly getActorExtractModelInputs: AgentSacGetActorExtractModelInputsCallback<TensorsIn>;
  readonly getActorInputTensors: AgentSacGetActorInputTensorsCallback<TensorsIn>;
  readonly nActions: number;
  readonly name: string;
}) => {
  
  let outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(
    maybeConcatenateTensors(getActorInputTensors({tensorsIn})),
  );
  outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs);

  const mu = tf.layers.dense({units: nActions}).apply(outputs);
  const logStd = tf.layers.dense({units: nActions}).apply(outputs);

  assert(mu instanceof tf.SymbolicTensor && logStd instanceof tf.SymbolicTensor);

  return tf.model({
    inputs: getActorExtractModelInputs({tensorsIn}),
    outputs: [mu, logStd],
    name,
  });
};

export const createCritic = async <
  TensorsIn extends SymbolicTensors
>({
  actionInput,
  getActorExtractModelInputs,
  getActorInputTensors,
  name,
  tensorsIn,
}: {
  readonly actionInput: tf.SymbolicTensor;
  readonly getActorExtractModelInputs: AgentSacGetActorExtractModelInputsCallback<TensorsIn>;
  readonly getActorInputTensors: AgentSacGetActorInputTensorsCallback<TensorsIn>;
  readonly name: string;
  readonly tensorsIn: TensorsIn;
}): Promise<tf.LayersModel> => {
  const inputTensors = getActorInputTensors({tensorsIn});

  let outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(
    tf.layers.concatenate().apply([...inputTensors, actionInput]),
  );

  outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs);
  outputs = tf.layers.dense({units: 1}).apply(outputs);

  assert(outputs instanceof tf.SymbolicTensor);

  const modelInputs = getActorExtractModelInputs({tensorsIn});

  const model = tf.model({
    inputs: [...modelInputs, actionInput],
    outputs,
    name,
  });

  model.trainable = true;
  return model;
};

const createAgentSacInstanceProps = async <
  TensorsIn extends SymbolicTensors
>({
  actorName,
  agentSacProps: {
    batchSize = 1, 
    nActions = 3, // 3 - impuls, 3 - RGB color
    gamma = 0.99, // Discount factor (γ)
    rewardScale = 10,
  },
  getActorCreateTensorsIn,
  getActorExtractModelInputs,
  getActorInputTensors
}: {
  readonly actorName: string;
  readonly agentSacProps: Partial<AgentSacConstructorProps>;
  readonly getActorCreateTensorsIn: AgentSacGetActorCreateTensorsInCallback<TensorsIn>;
  readonly getActorExtractModelInputs: AgentSacGetActorExtractModelInputsCallback<TensorsIn>;
  readonly getActorInputTensors: AgentSacGetActorInputTensorsCallback<TensorsIn>;
}): Promise<AgentSacInstanceProps<TensorsIn>> => {
      
  const maybeSavedActor = await loadModelByName(actorName);

  const tensorsIn = getActorCreateTensorsIn(Object.create(null));

  const actor: tf.LayersModel = maybeSavedActor ?? (
    await createActor<TensorsIn>({
      tensorsIn,
      nActions,
      name: actorName,
      getActorExtractModelInputs,
      getActorInputTensors,
    })
  );

  return {
    batchSize,
    nActions,
    gamma,
    rewardScale,
    // https://github.com/rail-berkeley/softlearning/blob/13cf187cc93d90f7c217ea2845067491c3c65464/softlearning/algorithms/sac.py#L37
    targetEntropy: -nActions,
    tensorsIn,
    actor,
  };
};

const createLogAlpha = ({
  logAlphaName,
}: {
  readonly logAlphaName: string;
}) => {
  const model = tf.sequential({name: logAlphaName});
  model.add(tf.layers.dense({units: 1, inputShape: [1], useBias: false}));
  model.setWeights([tf.tensor([0.0], [1, 1])]);
  return model;
};

const createAgentSacTrainableInstanceProps = async <
  TensorsIn extends SymbolicTensors
>({ 
  actorName,
  agentSacProps,
  getActorCreateTensorsIn,
  getActorExtractModelInputs,
  getActorInputTensors,
  logAlphaName,
  q1Name,
  q1TargetName,
  q2Name,
  q2TargetName,
  tau,
}: {
  readonly actorName: string;
  readonly agentSacProps: Partial<AgentSacConstructorProps>;
  readonly getActorCreateTensorsIn: AgentSacGetActorCreateTensorsInCallback<TensorsIn>;
  readonly getActorExtractModelInputs: AgentSacGetActorExtractModelInputsCallback<TensorsIn>;
  readonly getActorInputTensors: AgentSacGetActorInputTensorsCallback<TensorsIn>;
  readonly logAlphaName: string;
  readonly q1Name: string;
  readonly q1TargetName: string;
  readonly q2Name: string;
  readonly q2TargetName: string;
  readonly tau: number;
}): Promise<AgentSacTrainableInstanceProps<TensorsIn>> => {
  const agentSacInstanceProps =
    await createAgentSacInstanceProps({
      actorName,
      agentSacProps,
      getActorCreateTensorsIn,
      getActorExtractModelInputs,
      getActorInputTensors,
    });
  
  const {actor, nActions, tensorsIn} = agentSacInstanceProps;

  actor.trainable = true;

  const actorOptimizer = tf.train.adam();
  const actionInput = tf.input({batchShape: [null, nActions]});

  const createCriticByName = async (criticName: string) => {
    const maybeCritic = await loadModelByName(criticName);
    if (maybeCritic) return maybeCritic;

    return createCritic<TensorsIn>({
      actionInput,
      getActorExtractModelInputs,
      getActorInputTensors,
      name: criticName,
      tensorsIn,
    });
  };

  const [
    q1,
    q1Targ,
    q2,
    q2Targ,
    maybeLogAlpha,
  ] = await Promise.all([
    createCriticByName(q1Name),
    createCriticByName(q1TargetName),
    createCriticByName(q2Name),
    createCriticByName(q2TargetName),
    loadModelByName(logAlphaName),
  ]);

  const logAlphaModel = maybeLogAlpha || createLogAlpha({logAlphaName});
  const logAlpha = getLogAlphaByModel(logAlphaModel);

  const q1Optimizer = tf.train.adam();
  const q2Optimizer = tf.train.adam();
  const alphaOptimizer = tf.train.adam();

  return {
    ...agentSacInstanceProps,
    actorOptimizer,
    actionInput,
    alphaOptimizer,
    q1,
    q1Targ,
    q1Optimizer,
    q2,
    q2Targ,
    q2Optimizer,
    logAlpha,
    logAlphaModel,
    tau,
  };
};

const getLogAlphaByModel = (model: tf.LayersModel): tf.Variable<tf.Rank.R0> => {
  const [weights] = model.getWeights();
  assert(weights);

  const arraySync = weights.arraySync();
  assert(Array.isArray(arraySync));

  const [children] = arraySync;
  assert(Array.isArray(children));

  const [child] = children;
  assert(typeof child === 'number');

  return tf.variable(tf.scalar(child), true /* trainable */);
};

const updateTrainableTargets = ({
  q1,
  q1Targ,
  q2,
  q2Targ,
  tau,
}: {
  readonly q1: tf.LayersModel;
  readonly q1Targ: tf.LayersModel;
  readonly q2: tf.LayersModel;
  readonly q2Targ: tf.LayersModel;
  readonly tau: number;
}) => {
  const q1W = q1.getWeights();
  const q2W = q2.getWeights();

  const q1WTarg = q1Targ.getWeights();
  const q2WTarg = q2Targ.getWeights();

  const len = q1W.length;

  const tau_t = tf.scalar(tau);

  const calc = (w: tf.Tensor, wTarg: tf.Tensor) =>
    wTarg.mul(tf.scalar(1).sub(tau_t)).add(w.mul(tau_t));
    
  const w1 = [];
  const w2 = [];
  for (let i = 0; i < len; i++) {
    w1.push(calc(q1W[i], q1WTarg[i]));
    w2.push(calc(q2W[i], q2WTarg[i]));
  }

  q1Targ.setWeights(w1);
  q2Targ.setWeights(w2);
};

const saveCheckpoints = async ({
  actor,
  q1,
  q1Targ,
  q2,
  q2Targ,
  logAlpha,
  logAlphaModel,
}: {
  readonly actor: tf.LayersModel;
  readonly q1: tf.LayersModel;
  readonly q2: tf.LayersModel;
  readonly q1Targ: tf.LayersModel;
  readonly q2Targ: tf.LayersModel;
  readonly logAlpha: tf.Variable<tf.Rank.R0>;
  readonly logAlphaModel: tf.LayersModel;
}) => {
  void logAlphaModel.setWeights([tf.tensor([logAlpha.arraySync()], [1, 1])]);

  return Promise.all([
    saveModel(logAlphaModel),
    saveModel(actor),
    saveModel(q1),
    saveModel(q2),
    saveModel(q1Targ),
    saveModel(q2Targ),
  ]);
};

const trainAlpha = async ({
  actor,
  alphaOptimizer,
  batchSize,
  getPredictionArgs,
  logAlpha,
  state,
  targetEntropy,
}: {
  readonly actor: tf.LayersModel;
  readonly alphaOptimizer: tf.Optimizer;
  readonly batchSize: number;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
  readonly logAlpha: tf.Variable<tf.Rank.R0>;
  readonly state: tf.Tensor[];
  readonly targetEntropy: number;
}) => {
  const alphaLossFunction = (): tf.Scalar => {
    const sampledAction =
      sampleActionFrom({actor, batchSize, getPredictionArgs, state});

    assert(Array.isArray(sampledAction));
    const [, logPi] = sampledAction;

    const alpha = tf.exp(logAlpha);
    const loss = tf.scalar(-1).mul(alpha.mul(logPi.add(tf.scalar(targetEntropy))));

    assertShape(loss, [batchSize, 1]);
    return tf.mean(loss);
  };
  
  const {grads} = tf.variableGrads(alphaLossFunction, [logAlpha]);
  void alphaOptimizer.applyGradients(grads);
};

const trainActor = ({
  actor,
  actorOptimizer,
  batchSize,
  getPredictionArgs,
  logAlpha,
  nActions,
  q1,
  q2,
  state,
}: {
  readonly actor: tf.LayersModel;
  readonly actorOptimizer: tf.Optimizer;
  readonly batchSize: number;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
  readonly logAlpha: tf.Variable<tf.Rank.R0>;
  readonly nActions: number;
  readonly q1: tf.LayersModel;
  readonly q2: tf.LayersModel;
  readonly state: tf.Tensor[];
}) => {
  // TODO: consider delayed update of policy and targets (if possible)
  const actorLossFunction = (): tf.Scalar => {
    const sampledAction =
      sampleActionFrom({actor, batchSize, getPredictionArgs, state});
  
      assert(Array.isArray(sampledAction));
  
      const [freshAction, logPi] = sampledAction;

      const basePredictionArgs = getPredictionArgs({state});
      const nextPredictionArgs: tf.Tensor[] =
        Array.isArray(basePredictionArgs)
          ? [...basePredictionArgs, freshAction]
          : [basePredictionArgs, freshAction];
      
      const q1Value = q1.predict(nextPredictionArgs, {batchSize});
      const q2Value = q2.predict(nextPredictionArgs, {batchSize});
      
      assert(!Array.isArray(q1Value) && !Array.isArray(q2Value));
      const criticValue = tf.minimum(q1Value, q2Value);
  
      const alpha = tf.exp(logAlpha);
      const loss = alpha.mul(logPi).sub(criticValue);
  
      assertShape(freshAction, [batchSize, nActions]);
      assertShape(logPi, [batchSize, 1]);
      assertShape(q1Value, [batchSize, 1]);
      assertShape(criticValue, [batchSize, 1]);
      assertShape(loss, [batchSize, 1]);
  
      return tf.mean(loss)
  };
      
  const {grads} = tf.variableGrads(actorLossFunction, getTrainableOnlyWeights(actor));
  void actorOptimizer!.applyGradients(grads);
};

const trainCritics = ({
  actor,
  batchSize,
  gamma,
  getPredictionArgs,
  logAlpha,
  nActions,
  q1,
  q1Optimizer,
  q1Targ,
  q2,
  q2Optimizer,
  q2Targ,
  rewardScale,
  vectorizedTransitions: {state, action, reward, nextState},
}: {
  readonly actor: tf.LayersModel;
  readonly batchSize: number;
  readonly gamma: number;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
  readonly logAlpha: tf.Variable<tf.Rank.R0>;
  readonly nActions: number;
  readonly q1: tf.LayersModel;
  readonly q1Optimizer: tf.Optimizer;
  readonly q1Targ: tf.LayersModel;
  readonly q2: tf.LayersModel;
  readonly q2Optimizer: tf.Optimizer;
  readonly q2Targ: tf.LayersModel;
  readonly rewardScale: number;
  readonly vectorizedTransitions: VectorizedTransitions;
}) => {
  const getQLossFunction = (() => {

    const sampledAction =
      sampleActionFrom({actor, batchSize, getPredictionArgs, state});

    assert(Array.isArray(sampledAction));
    const [nextFreshAction, logPi] = sampledAction;

    const basePredictionArgs = getPredictionArgs({state: nextState});
    const nextPredictionArgs: tf.Tensor[] =
      Array.isArray(basePredictionArgs)
        ? [...basePredictionArgs, nextFreshAction]
        : [basePredictionArgs, nextFreshAction];

    const q1TargValue = q1Targ.predict(nextPredictionArgs, {batchSize});
    const q2TargValue = q2Targ.predict(nextPredictionArgs, {batchSize});
    
    assert(!Array.isArray(q1TargValue) && !Array.isArray(q2TargValue));
    const qTargValue = tf.minimum(q1TargValue, q2TargValue)
  
    // y = r + γ*(1 - d)*(min(Q1Targ(s', a'), Q2Targ(s', a')) - α*log(π(s'))
    const alpha = tf.exp(logAlpha);
    const target = reward.mul(tf.scalar(rewardScale)).add(tf.scalar(gamma).mul(qTargValue.sub(alpha.mul(logPi))));
                      
    assertShape(nextFreshAction, [batchSize, nActions]);
    assertShape(logPi, [batchSize, 1]);
    assertShape(qTargValue, [batchSize, 1]);
    assertShape(target, [batchSize, 1]);
  
    return (q: tf.LayersModel) => (): tf.Scalar => {
      const predictionArgs = getPredictionArgs({state});
      const qValue = q.predict(
        Array.isArray(predictionArgs) ? [...predictionArgs, action] : [predictionArgs, action],
        {batchSize},
      );

      assert(!Array.isArray(qValue));
      
      const loss = tf.scalar(0.5).mul(tf.mean(qValue.sub(target).square()));
      assertShape(qValue, [batchSize, 1]);

      return assertScalar(loss);
    };
  })();
  
  for (const [q, optimizer] of [[q1, q1Optimizer] as const, [q2, q2Optimizer] as const]) {
    assert(q && optimizer);
    const qLossFunction = getQLossFunction(q);
    const {grads} = tf.variableGrads(qLossFunction, getTrainableOnlyWeights(q!));
    void optimizer.applyGradients(grads);
  }

};

const trainAgentSac = ({
  actor,
  actorOptimizer,
  alphaOptimizer,
  batchSize,
  gamma,
  getPredictionArgs,
  logAlpha,
  nActions,
  q1,
  q1Optimizer,
  q1Targ,
  q2,
  q2Optimizer,
  q2Targ,
  rewardScale,
  targetEntropy,
  tau,
  transitions,
}: {
  readonly actor: tf.LayersModel;
  readonly actorOptimizer: tf.Optimizer;
  readonly alphaOptimizer: tf.Optimizer;
  readonly batchSize: number;
  readonly gamma: number;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
  readonly logAlpha: tf.Variable<tf.Rank.R0>;
  readonly nActions: number;
  readonly q1: tf.LayersModel;
  readonly q1Optimizer: tf.Optimizer;
  readonly q1Targ: tf.LayersModel;
  readonly q2: tf.LayersModel;
  readonly q2Optimizer: tf.Optimizer;
  readonly q2Targ: tf.LayersModel;
  readonly rewardScale: number;
  readonly targetEntropy: number;
  readonly tau: number;
  readonly transitions: readonly Omit<Transition, 'id' | 'priority'>[];
}) => tf.tidy(() => {

  const framesL: tf.Tensor[] = [];
  const framesR: tf.Tensor[] = [];
  const telemetries: tf.Tensor[] = [];
  const actions: tf.Tensor[] = [];
  const rewards: tf.Tensor[] = [];
  const nextFramesL: tf.Tensor[] = [];
  const nextFramesR: tf.Tensor[] = [];
  const nextTelemetries: tf.Tensor[] = [];
    
  for (const {
    state: [telemetry, frameL, frameR], 
    action, 
    reward, 
    nextState: [nextTelemetry, nextFrameL, nextFrameR] 
  } of transitions) {
    framesL.push(frameL);
    framesR.push(frameR);
    telemetries.push(telemetry);
    actions.push(action);
    rewards.push(reward);
    nextFramesL.push(nextFrameL);
    nextFramesR.push(nextFrameR);
    nextTelemetries.push(nextTelemetry);
  }

  // TODO: refactor to represent vectorized transitions
  const vectorizedTransitions: VectorizedTransitions = {
    state: [
      tf.stack(telemetries),
      tf.stack(framesL),
      tf.stack(framesR),
    ],
    action: tf.stack(actions), 
    reward: tf.stack(rewards), 
    nextState: [
      tf.stack(nextTelemetries),
      tf.stack(nextFramesL),
      tf.stack(nextFramesR),
    ],
  };
 
  void trainCritics({
    actor,
    batchSize,
    gamma,
    getPredictionArgs,
    logAlpha,
    nActions,
    q1,
    q1Optimizer,
    q1Targ,
    q2,
    q2Optimizer,
    q2Targ,
    rewardScale,
    vectorizedTransitions,
  });

  const {state} = vectorizedTransitions; 

  void trainActor({actor, actorOptimizer, batchSize, getPredictionArgs, logAlpha, nActions, q1, q2, state});
  void trainAlpha({actor, alphaOptimizer, batchSize, getPredictionArgs, logAlpha, state, targetEntropy});

  void updateTrainableTargets({q1, q1Targ, q2, q2Targ, tau});
});

const createAgentSacInstanceResult = <
  TensorsIn extends SymbolicTensors
>({
  agentSacInstanceProps,
  getPredictionArgs,
}: {
  readonly agentSacInstanceProps: AgentSacInstanceProps<TensorsIn>;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
}): AgentSacInstance => {
  const {actor, batchSize, nActions} = agentSacInstanceProps;

  const sampleAction: AgentSacSampleActionCallback =
    ({state}: AgentSacSampleActionCallbackProps) => sampleActionFrom({
      actor,
      batchSize,
      getPredictionArgs,
      state,
    }); 

  return {actor, batchSize, nActions, sampleAction};
};

export const createAgentSacInstance = async<
  TensorsIn extends SymbolicTensors
>({
  actorName,
  agentSacProps,
  getActorCreateTensorsIn,
  getActorExtractModelInputs,
  getActorInputTensors,
  getPredictionArgs,
}: {
  readonly actorName: string;
  readonly agentSacProps: Partial<AgentSacConstructorProps>;
  readonly getActorCreateTensorsIn: AgentSacGetActorCreateTensorsInCallback<TensorsIn>;
  readonly getActorExtractModelInputs: AgentSacGetActorExtractModelInputsCallback<TensorsIn>;
  readonly getActorInputTensors: AgentSacGetActorInputTensorsCallback<TensorsIn>;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
}): Promise<AgentSacInstance> => createAgentSacInstanceResult({
  agentSacInstanceProps: 
    await createAgentSacInstanceProps({
      actorName,
      agentSacProps,
      getActorCreateTensorsIn,
      getActorExtractModelInputs,
      getActorInputTensors,
    }),
  getPredictionArgs,
});

export const createAgentSacTrainableInstance = async <
  TensorsIn extends SymbolicTensors
>({
  actorName,
  agentSacProps,
  getActorCreateTensorsIn,
  getActorExtractModelInputs,
  getActorInputTensors,
  getPredictionArgs,
  logAlphaName,
  q1Name,
  q1TargetName,
  q2Name,
  q2TargetName,
  tau,
}: {
  readonly actorName: string;
  readonly agentSacProps: Partial<AgentSacConstructorProps>;
  readonly getActorCreateTensorsIn: AgentSacGetActorCreateTensorsInCallback<TensorsIn>;
  readonly getActorExtractModelInputs: AgentSacGetActorExtractModelInputsCallback<TensorsIn>;
  readonly getActorInputTensors: AgentSacGetActorInputTensorsCallback<TensorsIn>;
  readonly getPredictionArgs: AgentSacGetPredictionArgsCallback;
  readonly logAlphaName: string;
  readonly q1Name: string;
  readonly q1TargetName: string;
  readonly q2Name: string;
  readonly q2TargetName: string;
  readonly tau: number;
}): Promise<AgentSacTrainableInstance> => {
  const agentSacTrainableInstanceProps =
    await createAgentSacTrainableInstanceProps({
      actorName,
      agentSacProps,
      getActorCreateTensorsIn,
      getActorExtractModelInputs,
      getActorInputTensors,
      logAlphaName,
      q1Name,
      q1TargetName,
      q2Name,
      q2TargetName,
      tau,
    });

  void updateTrainableTargets(agentSacTrainableInstanceProps);

  const checkpoint: AgentSacTrainableCheckpointCallback =
    async () => void saveCheckpoints(agentSacTrainableInstanceProps);
  
  const agentSacInstanceResult =
    createAgentSacInstanceResult({
      agentSacInstanceProps: agentSacTrainableInstanceProps,
      getPredictionArgs,
    });

  const train: AgentSacTrainableTrainCallback = ({
    transitions,
  }: AgentSacTrainableTrainCallbackProps) => trainAgentSac({
    ...agentSacTrainableInstanceProps,
    getPredictionArgs,
    transitions,
  });

  return {...agentSacInstanceResult, checkpoint, train};
};
