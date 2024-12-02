import { Callable } from "./utils.js";
import {
  LogitsProcessor,
  PreTrainedTokenizer,
  Tensor,
} from "@huggingface/transformers";

interface TensorLike {
  data: Float32Array | Int32Array | number[];
  dims: number[];
  shape: number[];
  get(index: number): number;
  slice(start: number, end?: number): TensorLike;
  length: number;
}

export class StringStoppingCriteria extends Callable {
  constructor(
    private tokenizer: PreTrainedTokenizer,
    public promptLength: number,
  ) {
    super();
  }

  _call(inputIds: bigint[][], _scores?: TensorLike): boolean {
    const lastTokenId = Number(inputIds[0][inputIds[0].length - 1]);
    const lastToken = this.tokenizer.decode([lastTokenId], {
      skip_special_tokens: true,
    });

    return lastToken.includes('"');
  }
}

export class NumberStoppingCriteria extends Callable {
  constructor(
    private tokenizer: PreTrainedTokenizer,
    private promptLength: number,
    private precision: number = 3,
  ) {
    super();
  }

  _call(inputIds: TensorLike, _scores?: TensorLike): boolean {
    const relevantIds = [];
    for (let i = this.promptLength; i < inputIds.length; i++) {
      relevantIds.push(inputIds.get(i));
    }

    const decoded = this.tokenizer.decode(relevantIds, {
      skip_special_tokens: true,
    });

    if ((decoded.match(/\./g) || []).length > 1) {
      return true;
    }

    if (
      decoded.includes(".") &&
      decoded.trim().split(".")[1].length > this.precision
    ) {
      return true;
    }

    if (
      decoded.length > 1 &&
      /\d/.test(decoded) &&
      [" ", "\n"].includes(decoded[decoded.length - 1])
    ) {
      return true;
    }

    return false;
  }
}

export class OutputNumbersTokens extends LogitsProcessor {
  private allowedMask: boolean[];
  private vocabSize: number;

  constructor(
    private tokenizer: PreTrainedTokenizer,
    _prompt: string,
  ) {
    super();
    this.vocabSize = this.tokenizer.model.config.vocab_size;
    this.allowedMask = new Array(this.vocabSize).fill(false);

    for (let tokenId = 0; tokenId < this.vocabSize; tokenId++) {
      const tokenStr = this.tokenizer
        .decode([tokenId], {
          skip_special_tokens: true,
        })
        .trim();

      if (
        tokenStr === "" ||
        (!/[^\d.]/.test(tokenStr) && (tokenStr.match(/\./g) || []).length <= 1)
      ) {
        this.allowedMask[tokenId] = true;
      }
    }
  }

  _call(inputIds: bigint[][], logits: Tensor): Tensor {
    const logitsData = new Float32Array(logits.data);
    const shape = logits.dims;

    for (let batchIdx = 0; batchIdx < inputIds.length; batchIdx++) {
      for (let vocabIdx = 0; vocabIdx < this.vocabSize; vocabIdx++) {
        if (!this.allowedMask[vocabIdx]) {
          logitsData[batchIdx * this.vocabSize + vocabIdx] = -Infinity;
        }
      }
    }

    return new Tensor(logits.type, logitsData, shape);
  }
}
