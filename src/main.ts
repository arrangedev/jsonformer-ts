import {
  PreTrainedModel,
  PreTrainedTokenizer,
  Tensor,
} from "@huggingface/transformers";
import {
  NumberStoppingCriteria,
  OutputNumbersTokens,
  StringStoppingCriteria,
} from "./logits-processors.js";

export interface JsonSchema {
  type: string;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
}

interface JsonformerOptions {
  debug?: boolean;
  maxArrayLength?: number;
  maxNumberTokens?: number;
  temperature?: number;
  maxStringTokenLength?: number;
}

type GenerationValue = string | number | boolean | Record<string, any> | any[];

export class Jsonformer {
  private value: Record<string, any> = {};
  private numberLogitProcessor: OutputNumbersTokens;
  private readonly generationMarker: string = "|GENERATION|";
  private readonly options: Required<JsonformerOptions>;

  constructor(
    private model: PreTrainedModel,
    private tokenizer: PreTrainedTokenizer,
    private jsonSchema: JsonSchema,
    private prompt: string,
    options: JsonformerOptions = {},
  ) {
    this.options = {
      debug: false,
      maxArrayLength: 10,
      maxNumberTokens: 6,
      temperature: 1.0,
      maxStringTokenLength: 10,
      ...options,
    };

    this.numberLogitProcessor = new OutputNumbersTokens(
      this.tokenizer,
      this.prompt,
    );
  }

  private debug(
    caller: string,
    value: string,
    isPrompt: boolean = false,
  ): void {
    if (this.options.debug) {
      if (isPrompt) {
        console.log(`\x1b[32m${caller}\x1b[33m ${value}\x1b[0m`);
      } else {
        console.log(`\x1b[32m${caller}\x1b[34m ${value}\x1b[0m`);
      }
    }
  }

  private async generateNumber(
    temperature?: number,
    iterations: number = 0,
  ): Promise<number> {
    const prompt = this.getPrompt();
    this.debug("[generate_number]", prompt, true);

    const inputTokens = this.tokenizer.encode(prompt, {
      add_special_tokens: true,
    });

    const inputData = inputTokens.map(BigInt);
    const inputLength = inputTokens.length;
    const inputTensor = new Tensor("int64", inputData, [1, inputTokens.length]);

    const attentionMask = new Tensor(
      "int64",
      new Array(inputLength).fill(BigInt(1)),
      [1, inputLength],
    );
    const positionIds = new Tensor(
      "int64",
      Array.from({ length: inputLength }, (_, i) => BigInt(i)),
      [1, inputLength],
    );

    const stoppingCriteria = new NumberStoppingCriteria(
      this.tokenizer,
      inputLength,
    );

    const stopOnNumber = (inputIds: bigint[][], _scores?: number[][]) => {
      if (inputIds[0].length <= inputLength) {
        return false;
      }

      const tokenIds = inputIds[0].slice(inputLength);
      const decoded = this.tokenizer.decode(
        Array.from(tokenIds).map((t) => Number(t)),
        { skip_special_tokens: true },
      );

      // Check for multiple decimal points
      if ((decoded.match(/\./g) || []).length > 1) {
        return true;
      }

      // Stop if we have a number followed by space/newline
      if (
        decoded.length > 1 &&
        /\d/.test(decoded) &&
        [" ", "\n"].includes(decoded[decoded.length - 1])
      ) {
        return true;
      }

      return false;
    };

    const response = await this.model.generate({
      inputs: inputTensor,
      attention_mask: attentionMask,
      position_ids: positionIds,
      generation_config: {
        max_new_tokens: this.options.maxStringTokenLength,
        num_return_sequences: 1,
        pad_token_id: this.tokenizer.pad_token_id,
        temperature: temperature || this.options.temperature,
      },
      logits_processor: [this.numberLogitProcessor],
      stopping_criteria: [stopOnNumber],
    });

    let generatedTokens: bigint[];

    if ("data" in response) {
      generatedTokens = Array.from(response.data);
    } else {
      //@ts-ignore
      generatedTokens = Array.from(response.output_ids.data);
    }

    if (
      generatedTokens.length >= inputTokens.length &&
      generatedTokens
        .slice(0, inputTokens.length)
        .every((token, i) => token === BigInt(inputTokens[i]))
    ) {
      generatedTokens = generatedTokens.slice(inputTokens.length);
    }

    const result = this.tokenizer.decode(generatedTokens, {
      skip_special_tokens: true,
    });

    const processedResult = result.trim().replace(/\.$/, "");
    this.debug("[generate_number]", processedResult);

    try {
      return parseFloat(processedResult);
    } catch (error) {
      if (iterations > 3) {
        throw new Error("Failed to generate a valid number");
      }
      return this.generateNumber(
        (this.options.temperature || 1.0) * 1.3,
        iterations + 1,
      );
    }
  }

  private async generateBoolean(): Promise<boolean> {
    const prompt = this.getPrompt();
    this.debug("[generate_boolean]", prompt, true);

    const inputTokens = this.tokenizer.encode(prompt, {
      add_special_tokens: true,
      return_token_type_ids: true,
    });

    const inputData = inputTokens.map(BigInt);
    const inputTensor = new Tensor("int64", inputData, [1, inputTokens.length]);

    const attentionMask = new Tensor(
      "int64",
      Array(inputTokens.length).fill(BigInt(1)),
      [1, inputTokens.length],
    );

    const output = await this.model.forward({
      input_ids: inputTensor,
      attention_mask: attentionMask,
    });

    const logits = output.logits;
    const lastTokenLogits = logits.data.slice(-logits.dims[2]);

    const trueTokens = this.tokenizer.encode("true", {
      add_special_tokens: false,
    });
    const falseTokens = this.tokenizer.encode("false", {
      add_special_tokens: false,
    });

    const trueLogit = lastTokenLogits[trueTokens[0]];
    const falseLogit = lastTokenLogits[falseTokens[0]];

    const result = trueLogit > falseLogit;
    this.debug("[generate_boolean]", String(result));

    return result;
  }

  private async generateString(): Promise<string> {
    const prompt = `${this.getPrompt()}"`;
    this.debug("[generate_string]", prompt, true);

    const inputTokens = this.tokenizer.encode(prompt, {
      add_special_tokens: false,
    });

    const inputData = inputTokens.map(BigInt);
    const inputLength = inputTokens.length;

    const inputTensor = new Tensor("int64", inputData, [1, inputTokens.length]);
    const attentionMask = new Tensor(
      "int64",
      new Array(inputLength).fill(BigInt(1)),
      [1, inputLength],
    );
    const positionIds = new Tensor(
      "int64",
      Array.from({ length: inputLength }, (_, i) => BigInt(i)),
      [1, inputLength],
    );

    const stoppingCriteria = new StringStoppingCriteria(
      this.tokenizer,
      inputLength,
    );

    const response = await this.model.generate({
      inputs: inputTensor,
      attention_mask: attentionMask,
      position_ids: positionIds,
      generation_config: {
        max_new_tokens: this.options.maxStringTokenLength,
        num_return_sequences: 1,
        pad_token_id: this.tokenizer.pad_token_id,
        temperature: this.options.temperature,
        do_sample: true,
      },
      stopping_criteria: [stoppingCriteria],
    });

    let generatedTokens: bigint[];

    if ("data" in response) {
      generatedTokens = Array.from(response.data);
    } else {
      //@ts-ignore
      generatedTokens = Array.from(response.output_ids.data);
    }

    if (
      generatedTokens.length >= inputTokens.length &&
      generatedTokens
        .slice(0, inputTokens.length)
        .every((token, i) => token === BigInt(inputTokens[i]))
    ) {
      generatedTokens = generatedTokens.slice(inputTokens.length);
    }

    const decodedResponse = this.tokenizer.decode(generatedTokens, {
      skip_special_tokens: true,
    });

    this.debug("[generate_string]", `|${decodedResponse}|`);

    if (!decodedResponse.includes('"')) {
      return decodedResponse.trim();
    }

    return decodedResponse.split('"')[0].trim();
  }

  private async generateObject(
    properties: Record<string, JsonSchema>,
    obj: Record<string, any>,
  ): Promise<Record<string, any>> {
    for (const [key, schema] of Object.entries(properties)) {
      this.debug("[generate_object]", `generating value for ${key}`);
      obj[key] = await this.generateValue(schema, obj, key);
    }
    return obj;
  }

  private async generateValue(
    schema: JsonSchema,
    obj: Record<string, any> | any[],
    key?: string,
  ): Promise<GenerationValue> {
    const schemaType = schema.type;

    const isArray = (obj: Record<string, any> | any[]): obj is any[] => {
      return Array.isArray(obj);
    };

    const setGenerationMarker = (
      obj: Record<string, any> | any[],
      key?: string,
    ) => {
      if (key) {
        if (!isArray(obj)) {
          obj[key] = this.generationMarker;
        }
      } else {
        if (isArray(obj)) {
          obj.push(this.generationMarker);
        }
      }
    };

    switch (schemaType) {
      case "number":
        setGenerationMarker(obj, key);
        return this.generateNumber();

      case "boolean":
        setGenerationMarker(obj, key);
        return this.generateBoolean();

      case "string":
        setGenerationMarker(obj, key);
        return this.generateString();

      case "array":
        if (!schema.items) {
          throw new Error("Array schema must have items defined");
        }
        const newArray: any[] = [];
        if (key && !isArray(obj)) {
          obj[key] = newArray;
        }
        return this.generateArray(schema.items, newArray);

      case "object":
        if (!schema.properties) {
          throw new Error("Object schema must have properties defined");
        }
        const newObj: Record<string, any> = {};
        if (key) {
          if (!isArray(obj)) {
            obj[key] = newObj;
          }
        } else {
          if (isArray(obj)) {
            obj.push(newObj);
          }
        }
        return this.generateObject(schema.properties, newObj);

      default:
        throw new Error(`Unsupported schema type: ${schemaType}`);
    }
  }

  private async generateArray(
    itemSchema: JsonSchema,
    obj: any[],
  ): Promise<any[]> {
    for (let i = 0; i < this.options.maxArrayLength; i++) {
      const element = await this.generateValue(itemSchema, obj);
      obj[obj.length - 1] = element;

      obj.push(this.generationMarker);
      const inputPrompt = this.getPrompt();
      obj.pop();

      const inputTokens = this.tokenizer.encode(inputPrompt, {
        add_special_tokens: true,
        return_token_type_ids: true,
      });

      const inputData = inputTokens.map(BigInt);
      const inputTensor = new Tensor("int64", inputData, [
        1,
        inputTokens.length,
      ]);

      const attentionMask = new Tensor(
        "int64",
        Array(inputTokens.length).fill(BigInt(1)),
        [1, inputTokens.length],
      );

      const output = await this.model.forward({
        input_ids: inputTensor,
        attention_mask: attentionMask,
      });

      const logits = output.logits;
      const lastTokenLogits = logits.data.slice(-logits.dims[2]);

      const topK = 30;
      const tokenScores = lastTokenLogits.map(
        (score: number, index: number) => ({
          score,
          index,
        }),
      );
      const sortedTokens = tokenScores
        .sort((a: any, b: any) => b.score - a.score)
        .slice(0, topK);

      let foundComma = false;
      let foundCloseBracket = false;

      for (const { index } of sortedTokens) {
        try {
          const tokenId = Math.floor(index);
          if (Number.isInteger(tokenId) && tokenId >= 0) {
            const decodedToken = this.tokenizer.decode([tokenId], {
              skip_special_tokens: false,
            });

            if (decodedToken.includes(",")) {
              foundComma = true;
              break;
            }
            if (decodedToken.includes("]")) {
              foundCloseBracket = true;
              break;
            }
          }
        } catch (error) {
          console.warn(`Failed to decode token at index ${index}`, error);
          continue;
        }
      }

      if (foundCloseBracket || !foundComma) {
        break;
      }
    }

    return obj;
  }

  private getPrompt(): string {
    return `${this.prompt}\nOutput result in the following JSON schema format:\n${JSON.stringify(this.jsonSchema)}\nResult: ${this.getProgress()}`;
  }

  private getProgress(): string {
    const progress = JSON.stringify(this.value);
    const genMarkerIndex = progress.indexOf(`"${this.generationMarker}"`);

    if (genMarkerIndex !== -1) {
      return progress.substring(0, genMarkerIndex);
    }

    throw new Error("Failed to find generation marker");
  }

  public async generate(): Promise<Record<string, any>> {
    this.value = {};
    if (!this.jsonSchema.properties) {
      throw new Error("Root schema must have properties defined");
    }

    const generatedData = await this.generateObject(
      this.jsonSchema.properties,
      this.value,
    );
    return generatedData;
  }
}

export type { JsonformerOptions };
export default Jsonformer;
