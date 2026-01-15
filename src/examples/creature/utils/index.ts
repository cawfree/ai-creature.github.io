import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {
  AgentSacConstructorProps,
  AgentSacGetActorCreateTensorsInCallback,
  AgentSacGetActorExtractModelInputsCallback,
  AgentSacGetActorInputTensorsCallback,
  AgentSacGetPredictionArgsCallback,
} from '../../../@types';
import {NAME} from '../../../constants';
import {
  createAgentSacInstance,
  createAgentSacTrainableInstance,
} from '../../../utils';

import {CreatureTensorsIn} from '../@types';

const frameStackShape: [number, number, number] = [25, 25, 3];
const padding = 'valid';
const kernelInitializer = 'glorotNormal';
const biasInitializer = 'glorotNormal';

const getPredictionArgs: AgentSacGetPredictionArgsCallback = ({state}) => state;

const createConvEncoder = (inputs: tf.SymbolicTensor): tf.SymbolicTensor => { 

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

const getActorExtractModelInputs: AgentSacGetActorExtractModelInputsCallback<CreatureTensorsIn> = ({
  tensorsIn: {
    frameInputL,
    frameInputR,
    telemetryInput,
  },
}) => [telemetryInput, frameInputL, frameInputR];

const getActorInputTensors: AgentSacGetActorInputTensorsCallback<CreatureTensorsIn> = ({
  tensorsIn: {
    frameInputL,
    frameInputR,
    telemetryInput,
  },
}) => [
  telemetryInput,
  createConvEncoder(frameInputL),
  createConvEncoder(frameInputR),
];

const getActorCreateTensorsIn: AgentSacGetActorCreateTensorsInCallback<CreatureTensorsIn> = ({
  nTelemetry,
}) => ({
  frameInputL: tf.input({batchShape : [null, ...frameStackShape]}),
  frameInputR: tf.input({batchShape : [null, ...frameStackShape]}),
  telemetryInput: tf.input({batchShape : [null, nTelemetry]}),
});

// trainAgent
// assertShape(state[0], [batchSize, nTelemetry]);
// assertShape(state[1], [batchSize, ...frameStackShape]);
// assertShape(action, [batchSize, nActions]);
// assertShape(reward, [batchSize, 1]);
// assertShape(nextState[0], [batchSize, nTelemetry]);
// assertShape(nextState[1], [batchSize, ...frameStackShape]);

export const createCreatureAgentSacInstance = async ({
  // TODO: force specify name
  actorName = NAME.ACTOR,
  agentSacProps = Object.create(null),
}: {
  readonly actorName?: string;
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
} = Object.create(null)) => {
  const agentSacInstance =
    await createAgentSacInstance({
      actorName,
      agentSacProps,
      getActorCreateTensorsIn,
      getPredictionArgs,
      getActorExtractModelInputs,
      getActorInputTensors,
    });

  return {...agentSacInstance, frameStackShape};
};

export const createCreatureAgentSacTrainableInstance = async ({
  // TODO: force specify names
  actorName = NAME.ACTOR,
  agentSacProps = Object.create(null),
  logAlphaName = NAME.ALPHA,
  q1Name = NAME.Q1,
  q1TargetName = NAME.Q1_TARGET,
  q2Name = NAME.Q2,
  q2TargetName = NAME.Q2_TARGET,
  tau = 5e-3,
}: {
  readonly actorName?: string;
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
  readonly logAlphaName?: string;
  readonly q1Name?: string;
  readonly q1TargetName?: string;
  readonly q2Name?: string;
  readonly q2TargetName?: string;
  readonly tau?: number;
} = Object.create(null)) => {
  const agentSacTrainableInstance =
    await createAgentSacTrainableInstance({
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
    });

  return {...agentSacTrainableInstance, frameStackShape};
};
