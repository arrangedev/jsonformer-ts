import { PreTrainedModel, PreTrainedTokenizer } from "@xenova/transformers";
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

    const inputTokens = this.tokenizer.encode(prompt, null, {
      add_special_tokens: true,
    });

    const response = await this.model.generate(inputTokens, {
      max_new_tokens: this.options.maxNumberTokens,
      num_return_sequences: 1,
      logits_processor: [this.numberLogitProcessor],
      stopping_criteria: [
        new NumberStoppingCriteria(this.tokenizer, inputTokens.length),
      ],
      temperature: temperature || this.options.temperature,
      pad_token_id: this.tokenizer.pad_token_id,
    });

    const result = this.tokenizer.decode(response, {
      skip_special_tokens: true,
    });

    const processedResult = result
      .slice(prompt.length)
      .trim()
      .replace(/\.$/, "");
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

    const inputTensor = this.tokenizer.encode(prompt, null, {
      add_special_tokens: true,
    });

    const output = await this.model.forward(inputTensor);
    const logits = output.logits.get(output.logits.length - 1);

    const trueTokens = this.tokenizer.encode("true", null, {
      add_special_tokens: false,
    });
    const falseTokens = this.tokenizer.encode("false", null, {
      add_special_tokens: false,
    });

    const result = logits.get(trueTokens[0]) > logits.get(falseTokens[0]);
    this.debug("[generate_boolean]", String(result));

    return result;
  }

  private async generateString(): Promise<string> {
    const prompt = `${this.getPrompt()}"`;
    this.debug("[generate_string]", prompt, true);

    const inputTokens = this.tokenizer.encode(prompt, null, {
      add_special_tokens: true,
    });

    const response = await this.model.generate(inputTokens, {
      max_new_tokens: this.options.maxStringTokenLength,
      num_return_sequences: 1,
      temperature: this.options.temperature,
      stopping_criteria: [
        new StringStoppingCriteria(this.tokenizer, inputTokens.length),
      ],
      pad_token_id: this.tokenizer.pad_token_id,
    });

    const decodedResponse = this.tokenizer.decode(response, {
      skip_special_tokens: true,
    });

    this.debug("[generate_string]", `|${decodedResponse}|`);

    if (!decodedResponse.includes('"')) {
      return decodedResponse;
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

      const inputTensor = this.tokenizer.encode(inputPrompt, null, {
        add_special_tokens: true,
      });

      const output = await this.model.forward(inputTensor);
      const logits = output.logits.get(output.logits.length - 1);

      const topK = 30;
      const topValues = new Array(topK).fill(-Infinity);
      const topIndices = new Array(topK).fill(0);

      for (let i = 0; i < logits.length; i++) {
        const value = logits.get(i);
        let pos = topK - 1;
        while (pos >= 0 && value > topValues[pos]) {
          if (pos < topK - 1) {
            topValues[pos + 1] = topValues[pos];
            topIndices[pos + 1] = topIndices[pos];
          }
          pos--;
        }
        if (pos < topK - 1) {
          topValues[pos + 1] = value;
          topIndices[pos + 1] = i;
        }
      }

      let foundComma = false;
      let foundCloseBracket = false;

      for (const tokenId of topIndices) {
        const decodedToken = this.tokenizer.decode([tokenId], {
          skip_special_tokens: true,
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