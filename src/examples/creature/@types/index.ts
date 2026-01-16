import * as tf from '@tensorflow/tfjs';
import { AgentSacInstance, SampledAction } from '../../../@types';

export type CreatureTensorsIn = {
  readonly frameInputL: tf.SymbolicTensor;
  readonly frameInputR: tf.SymbolicTensor;
  readonly telemetryInput: tf.SymbolicTensor;
};

export type Telemetry = {
  readonly lidar: number;
  readonly linearVelocityNormX: number;
  readonly linearVelocityNormY: number;
  readonly linearVelocityNormZ: number;
  readonly accelerationX: number;
  readonly accelerationY: number;
  readonly accelerationZ: number;
  readonly windowCollisionX: number;
  readonly windowCollisionY: number;
  readonly windowCollisionZ: number;
};

export type CreatureGetActionProps = {
  readonly imageLeft: HTMLImageElement;
  readonly imageRight: HTMLImageElement;
  readonly telemetry: Telemetry;
};

export type CreatureGetActionResult = {
  readonly imageLeftPixelsNorm: tf.Tensor;
  readonly imageRightPixelsNorm: tf.Tensor;
  readonly action: readonly number[];
  readonly telemetry: Telemetry;
};

export type CreatureGetActionCallback =
  (props: CreatureGetActionProps) => Promise<CreatureGetActionResult>;

export type CreatureCreateTransitionProps = {
  readonly imageLeftPixelsNorm: tf.Tensor3D;
  readonly imageRightPixelsNorm: tf.Tensor3D;
  readonly reward: number;
  readonly action: readonly number[];
  readonly telemetry: Telemetry;
};

// TODO: normalize this
export type NormalizedCreatureState = [telemetry: number[], framesArrL: number[][][], framesArrR: number[][][]];

export type SerializedTransition = {
  readonly action: readonly number[];
  readonly id: number;
  readonly reward: number;
  readonly state: NormalizedCreatureState;
};

export type CreatureCreateTransitionResult = {
  readonly nextTransition: SerializedTransition;
};

export type CreatureCreateTransitionCallback = (
  props: CreatureCreateTransitionProps
) => Promise<CreatureCreateTransitionResult>;

export type CreatureAgentSacInstance =
  & Omit<AgentSacInstance, 'sampleAction'>
  & {
    readonly createTransition: CreatureCreateTransitionCallback;
    readonly frameStackShape: [number, number, number];
    readonly getAction: CreatureGetActionCallback;
  };
