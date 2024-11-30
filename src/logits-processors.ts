import { PreTrainedTokenizer } from "@xenova/transformers";

interface TensorLike {
  data: Float32Array | Int32Array | number[];
  dims: number[];
  shape: number[];
  get(index: number): number;
  slice(start: number, end?: number): TensorLike;
  length: number;
}

export class StringStoppingCriteria {
  constructor(
    private tokenizer: PreTrainedTokenizer,
    private promptLength: number,
  ) {}

  invoke(inputIds: TensorLike, _scores?: TensorLike): boolean {
    if (inputIds.length <= this.promptLength) {
      return false;
    }

    const lastTokenId = inputIds.get(inputIds.length - 1);
    const lastToken = this.tokenizer.decode([lastTokenId], {
      skip_special_tokens: true,
    });

    return lastToken.includes('"');
  }
}

export class NumberStoppingCriteria {
  constructor(
    private tokenizer: PreTrainedTokenizer,
    private promptLength: number,
    private precision: number = 3,
  ) {}

  invoke(inputIds: TensorLike, _scores?: TensorLike): boolean {
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

export class OutputNumbersTokens {
  private allowedMask: boolean[];
  private vocabSize: number;

  constructor(
    private tokenizer: PreTrainedTokenizer,
    _prompt: string,
  ) {
    this.vocabSize = this.tokenizer.model.config.vocab_size;
    this.allowedMask = new Array(this.vocabSize).fill(false);

    const tokenIds = Array.from({ length: this.vocabSize }, (_, i) => i);

    tokenIds.forEach(async (tokenId) => {
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
    });
  }

  invoke(_: any, scores: TensorLike): TensorLike {
    const inputArray = Array.from(scores.data);
    const maskedScores = new Float32Array(inputArray.length);

    for (let i = 0; i < inputArray.length; i++) {
      maskedScores[i] = this.allowedMask[i] ? inputArray[i] : -Infinity;
    }

    return {
      data: maskedScores,
      dims: scores.dims,
      shape: scores.shape,
      get(index: number) {
        return this.data[index];
      },
      slice(start: number, end?: number) {
        const slicedData = Array.from(this.data).slice(start, end);
        return {
          data: new Float32Array(slicedData),
          dims: [slicedData.length],
          shape: [slicedData.length],
          get(index: number) {
            return this.data[index];
          },
          slice: this.slice,
          length: slicedData.length,
        };
      },
      length: maskedScores.length,
    };
  }
}
