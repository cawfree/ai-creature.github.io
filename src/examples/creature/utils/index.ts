import assert from 'minimalistic-assert';
import * as tf from '@tensorflow/tfjs';

import {
  AgentSacConstructorProps,
  AgentSacGetActorCreateTensorsInCallback,
  AgentSacGetActorExtractModelInputsCallback,
  AgentSacGetActorInputTensorsCallback,
  VectorizedTransitions,
  VectorizeTransitionsCallback,
} from '../../../@types';
import {NAME} from '../../../constants';
import {
  createAgentSacInstance,
  createAgentSacTrainableInstance,
} from '../../../utils';

import {
  CreatureAgentSacInstance,
  CreatureCreateTransitionCallback,
  CreatureCreateTransitionProps,
  CreatureGetActionCallback,
  CreatureGetActionProps,
  CreatureGetActionResult,
  CreatureState,
  CreatureTensorsIn,
  NormalizedCreatureState,
  SerializedTransition,
  Telemetry,
} from '../@types';

const frameStackShape: [number, number, number] = [25, 25, 3];
const padding = 'valid';
const kernelInitializer = 'glorotNormal';
const biasInitializer = 'glorotNormal';
// 3 - linear valocity, 3 - acceleration, 3 - collision point, 1 - lidar (tanh of distance)
const nTelemetry = 10;

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

const getActorCreateTensorsIn: AgentSacGetActorCreateTensorsInCallback<CreatureTensorsIn> = () => ({
  telemetryInput: tf.input({batchShape : [null, nTelemetry]}),
  frameInputL: tf.input({batchShape : [null, ...frameStackShape]}),
  frameInputR: tf.input({batchShape : [null, ...frameStackShape]}),
});

const vectorizeTelemetry = ({
  linearVelocityNormX,
  linearVelocityNormY,
  linearVelocityNormZ,
  accelerationX,
  accelerationY,
  accelerationZ,
  windowCollisionX,
  windowCollisionY,
  windowCollisionZ,
  lidar,
}: Telemetry) => [
  linearVelocityNormX,
  linearVelocityNormY,
  linearVelocityNormZ,
  accelerationX,
  accelerationY,
  accelerationZ,
  windowCollisionX,
  windowCollisionY,
  windowCollisionZ,
  lidar,
];

export const createCreatureAgentSacInstance = async ({
  // TODO: force specify name
  actorName = NAME.ACTOR,
  agentSacProps = Object.create(null),
}: {
  readonly actorName?: string;
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
} = Object.create(null)): Promise<CreatureAgentSacInstance> => {
  const {sampleAction, nActions, ...extras} =
    await createAgentSacInstance({
      actorName,
      agentSacProps,
      getActorCreateTensorsIn,
      getActorExtractModelInputs,
      getActorInputTensors,
    });
  
  const getAction: CreatureGetActionCallback = async ({
    imageLeft,
    imageRight,
    telemetry,
  }: CreatureGetActionProps): Promise<CreatureGetActionResult> => {
    const [
      imageLeftPixelsNorm,
      imageRightPixelsNorm,
    ] = tf.tidy(() => {
      const imageLeftPixels = tf.browser.fromPixels(imageLeft);
      const imageRightPixels = tf.browser.fromPixels(imageRight);
      return [
        tf.concat([imageLeftPixels.sub(255/2).div(255/2)], -1) /* resL */,
        tf.concat([imageRightPixels.sub(255/2).div(255/2)], -1) /* resR */,
      ];
    });

    const imageLeftFrame = tf.stack([imageLeftPixelsNorm]);
    const imageRightFrame = tf.stack([imageRightPixelsNorm]);
    const telemetryBatch = tf.tensor(vectorizeTelemetry(telemetry), [1, nTelemetry])

    const [pi, logPi] = sampleAction({
      state: [telemetryBatch, imageLeftFrame, imageRightFrame],
    });

    const piArray = await pi.array();
    assert(Array.isArray(piArray));

    const maybeAction = piArray[0];
    assert(Array.isArray(maybeAction));

    const action = maybeAction.flatMap((e: unknown) => typeof e === 'number' ? [e] : []);
    assert(action.length === nActions);

    void imageLeftFrame.dispose();
    void imageRightFrame.dispose();
    void telemetryBatch.dispose();

    void pi.dispose();
    void logPi.dispose();

    return {
      // NOTE: The caller is expected to take ownership
      //       of disposing these tensors.
      imageLeftPixelsNorm,
      imageRightPixelsNorm,
      action,
      telemetry,
    };
  };

  let stateId = 0;

  const createTransition: CreatureCreateTransitionCallback = async({
    imageLeftPixelsNorm,
    imageRightPixelsNorm,
    reward,
    action: actionArr,
    telemetry,
  }: CreatureCreateTransitionProps)  => {

    const [framesArrL, framesArrR] = await Promise.all([
      imageLeftPixelsNorm.array(),
      imageRightPixelsNorm.array(),
    ]);

    const normalizedState: NormalizedCreatureState = [
      vectorizeTelemetry(telemetry),
      framesArrL,
      framesArrR,
    ];
    
    // TODO: strong typing for serialized transitions
    const nextTransition: SerializedTransition = {
      id: stateId++, 
      state: normalizedState,
      action: actionArr,
      reward,
    };

    return {
      nextTransition,
      imageLeftPixelsNorm,
      imageRightPixelsNorm,
    };
  };

  return {
    ...extras,
    nActions,
    createTransition,
    frameStackShape,
    getAction,
  };
};

const vectorizeTransitions: VectorizeTransitionsCallback<CreatureState> = ({
  transitions,
}): VectorizedTransitions => {
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
    void framesL.push(frameL);
    void framesR.push(frameR);
    void telemetries.push(telemetry);
    void actions.push(action);
    void rewards.push(reward);
    void nextFramesL.push(nextFrameL);
    void nextFramesR.push(nextFrameR);
    void nextTelemetries.push(nextTelemetry);
  }

  return {
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
      logAlphaName,
      q1Name,
      q1TargetName,
      q2Name,
      q2TargetName,
      tau,
      vectorizeTransitions,
    });

  return {...agentSacTrainableInstance, frameStackShape, nTelemetry};
};
