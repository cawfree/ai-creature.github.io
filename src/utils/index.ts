import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';
import isEqual from 'react-fast-compare';

import {AgentSacConstructorProps, AgentSacInstanceProps, AgentSacTrainableInstanceProps} from '../@types';
import {AgentSac, AgentSacTrainable} from '../classes';
import {NAME, VERSION} from '../constants';

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

export const getTrainableOnlyWeights = (layersModel: tf.LayersModel) =>
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

const createConvEncoder = (inputs: tf.SymbolicTensor): tf.SymbolicTensor => {
  const padding = 'valid';
  const kernelInitializer = 'glorotNormal';
  const biasInitializer = 'glorotNormal';

  let outputs = tf.layers.conv2d({
    filters: 16,
    kernelSize: 5,
    strides: 2,
    padding,
    kernelInitializer,
    biasInitializer,
    activation: 'relu',
    trainable: true,
  }).apply(inputs);

  outputs = tf.layers.maxPooling2d({poolSize:2}).apply(outputs);

  outputs = tf.layers.conv2d({
    filters: 16,
    kernelSize: 3,
    strides: 1,
    padding,
    kernelInitializer,
    biasInitializer,
    activation: 'relu',
    trainable: true,
  }).apply(outputs);

  outputs = tf.layers.maxPooling2d({poolSize:2}).apply(outputs);
  outputs = tf.layers.flatten().apply(outputs);

  assert(outputs instanceof tf.SymbolicTensor);
  return outputs;
};

export const createActor = async ({
  frameInputL,
  frameInputR,
  nActions,
  name,
  sighted,
  telemetryInput,
}: {
  readonly frameInputL: tf.SymbolicTensor;
  readonly frameInputR: tf.SymbolicTensor;
  readonly nActions: number;
  readonly name: string;
  readonly sighted: boolean;
  readonly telemetryInput: tf.SymbolicTensor;
}) => {
  let outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(
    sighted
      ? tf.layers.concatenate().apply([
          createConvEncoder(frameInputL),
          createConvEncoder(frameInputR),
          telemetryInput,
        ])
      : telemetryInput
  );

  outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs);

  const mu = tf.layers.dense({units: nActions}).apply(outputs);
  const logStd = tf.layers.dense({units: nActions}).apply(outputs);

  assert(mu instanceof tf.SymbolicTensor && logStd instanceof tf.SymbolicTensor);

  return tf.model({
    inputs: sighted
      ? [telemetryInput, frameInputL, frameInputR]
      : [telemetryInput],
      outputs: [mu, logStd],
      name,
  });
};

export const createCritic = async ({
  actionInput,
  frameInputL,
  frameInputR,
  name,
  sighted,
  telemetryInput,
}: {
  readonly actionInput: tf.SymbolicTensor;
  readonly frameInputL: tf.SymbolicTensor;
  readonly frameInputR: tf.SymbolicTensor;
  readonly name: string;
  readonly sighted: boolean;
  readonly telemetryInput: tf.SymbolicTensor;
}): Promise<tf.LayersModel> => {
  const base = tf.layers.concatenate().apply([telemetryInput, actionInput]);
  assert(base instanceof tf.SymbolicTensor);

  let outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(
    sighted
      ? tf.layers.concatenate().apply([
          createConvEncoder(frameInputL),
          createConvEncoder(frameInputR),
          base,
        ])
      : base
  );

  outputs = tf.layers.dense({units: 256, activation: 'relu'}).apply(outputs);
  outputs = tf.layers.dense({units: 1}).apply(outputs);

  assert(outputs instanceof tf.SymbolicTensor);

  const model = tf.model({
    inputs: sighted 
      ? [telemetryInput, frameInputL, frameInputR, actionInput] 
      : [telemetryInput, actionInput],
    outputs,
    name,
  });

  model.trainable = true;
  return model;
};

const createAgentSacInstanceProps = async ({
  actorName,
  agentSacProps: {
    batchSize = 1, 
    frameShape = [25, 25, 3], 
    nFrames = 1, // Number of stacked frames per state
    nActions = 3, // 3 - impuls, 3 - RGB color
    nTelemetry = 10, // 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
    gamma = 0.99, // Discount factor (γ)
    tau = 5e-3, // Target smoothing coefficient (τ)
    sighted = true,
    rewardScale = 10,
  },
}: {
  readonly actorName: string;
  readonly agentSacProps: Partial<AgentSacConstructorProps>;
}): Promise<AgentSacInstanceProps> => {
  const frameStackShape = [...frameShape.slice(0, 2), frameShape[2] * nFrames] as [number, number, number];

  const frameInputL = tf.input({batchShape : [null, ...frameStackShape]});
  const frameInputR = tf.input({batchShape : [null, ...frameStackShape]});
  const telemetryInput = tf.input({batchShape : [null, nTelemetry]});
      
  const maybeSavedActor = await loadModelByName(actorName);

  const actor: tf.LayersModel = maybeSavedActor ?? (
    await createActor({
      frameInputL,
      frameInputR,
      nActions,
      name: actorName,
      sighted,
      telemetryInput,
    })
  );

  return {
    batchSize,
    frameShape,
    nFrames,
    nActions,
    nTelemetry,
    gamma,
    tau,
    sighted,
    rewardScale,
    frameStackShape,
    // https://github.com/rail-berkeley/softlearning/blob/13cf187cc93d90f7c217ea2845067491c3c65464/softlearning/algorithms/sac.py#L37
    targetEntropy: -nActions,
    frameInputL,
    frameInputR,
    telemetryInput,
    actor,
  };
};

export const createAgentSac = async ({
  // TODO: force specify name
  actorName = NAME.ACTOR,
  agentSacProps = Object.create(null),
}: {
  readonly actorName?: string;
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
} = Object.create(null)) => {
  const agentSacInstanceProps =
    await createAgentSacInstanceProps({actorName, agentSacProps});

  const agent = new AgentSac(agentSacInstanceProps);
  // TODO: remove `Initializable`.
  await agent.initialize();
  return {agent};
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

export const createAgentSacTrainableInstanceProps = async ({ 
  actorName,
  agentSacProps,
  logAlphaName,
  q1Name,
  q1TargetName,
  q2Name,
  q2TargetName,
}: {
  readonly actorName: string;
  readonly agentSacProps: Partial<AgentSacConstructorProps>;
  readonly logAlphaName: string;
  readonly q1Name: string;
  readonly q1TargetName: string;
  readonly q2Name: string;
  readonly q2TargetName: string;
}): Promise<AgentSacTrainableInstanceProps> => {
  const agentSacInstanceProps =
    await createAgentSacInstanceProps({actorName, agentSacProps});
  
  const {
    actor,
    frameInputL,
    frameInputR,
    nActions,
    sighted,
    telemetryInput,
  } = agentSacInstanceProps;

  actor.trainable = true;

  const actorOptimizer = tf.train.adam();
  const actionInput = tf.input({batchShape: [null, nActions]});

  const createCriticByName = async (criticName: string) => {
    const maybeCritic = await loadModelByName(criticName);
    if (maybeCritic) return maybeCritic;

    return createCritic({
      actionInput,
      frameInputL,
      frameInputR,
      name: criticName,
      sighted,
      telemetryInput,
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
    logAlphaModel,
  };
};

export const createAgentSacTrainable = async ({
  // TODO: force specify names
  actorName = NAME.ACTOR,
  agentSacProps = Object.create(null),
  logAlphaName = NAME.ALPHA,
  q1Name = NAME.Q1,
  q1TargetName = NAME.Q1_TARGET,
  q2Name = NAME.Q2,
  q2TargetName = NAME.Q2_TARGET,
}: {
  readonly actorName?: string;
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
  readonly logAlphaName?: string;
  readonly q1Name?: string;
  readonly q1TargetName?: string;
  readonly q2Name?: string;
  readonly q2TargetName?: string;
} = Object.create(null)) => {
  const agentSacTrainableInstanceProps =
    await createAgentSacTrainableInstanceProps({
      actorName,
      agentSacProps,
      logAlphaName,
      q1Name,
      q1TargetName,
      q2Name,
      q2TargetName,
    });

  const agent = new AgentSacTrainable(agentSacTrainableInstanceProps);

  await agent.initialize();
  await agent.checkpoint();
  return {agent};
};

export const getLogAlphaByModel = (model: tf.LayersModel): tf.Variable<tf.Rank.R0> => {
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
