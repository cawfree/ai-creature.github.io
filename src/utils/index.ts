import * as tf from '@tensorflow/tfjs';
import assert from 'minimalistic-assert';
import isEqual from 'react-fast-compare';

import {AgentSacConstructorProps, AgentSacInstanceProps} from '../@types';
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

export const createAgentSacTrainable = async ({
  // TODO: force specify name
  actorName = NAME.ACTOR,
  agentSacProps = Object.create(null),
}: {
  readonly actorName?: string;
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
} = Object.create(null)) => {
  const agentSacInstanceProps =
    await createAgentSacInstanceProps({actorName, agentSacProps});
  const agent = new AgentSacTrainable(agentSacInstanceProps);
  await agent.initialize();
  await agent.checkpoint();
  return {agent};
};