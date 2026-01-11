import assert from 'minimalistic-assert';

export abstract class Initializable {

  #initialized: boolean;

  constructor() {
    this.#initialized = false;
  }

  async initialize(): Promise<void> {
    assert(!this.#initialized);
  }

  initialized(): boolean {
    return this.#initialized;
  }

}