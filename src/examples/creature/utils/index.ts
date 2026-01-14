import {
  AgentSacConstructorProps,
  AgentSacInstance,
} from '../../../@types';
import {NAME} from '../../../constants';
import {
  createAgentSacInstance,
  createAgentSacTrainableInstance,
} from '../../../utils';

export const createCreatureAgentSacInstance = ({
  // TODO: force specify name
  actorName = NAME.ACTOR,
  agentSacProps = Object.create(null),
}: {
  readonly actorName?: string;
  readonly agentSacProps?: Partial<AgentSacConstructorProps>;
} = Object.create(null)): Promise<AgentSacInstance> => createAgentSacInstance({
  actorName,
  agentSacProps,
});

export const createCreatureAgentSacTrainableInstance = ({
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
} = Object.create(null)) => createAgentSacTrainableInstance({
  actorName,
  agentSacProps,
  logAlphaName,
  q1Name,
  q1TargetName,
  q2Name,
  q2TargetName,
  tau,
});
