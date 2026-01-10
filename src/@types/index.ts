export type Transition = {
    readonly id: number;
    readonly priority: number;
    readonly state: unknown[];
    readonly action: unknown
    readonly reward: number;
};
