import ora from "ora";
import {
  AutoModel,
  AutoModelForCausalLM,
  AutoTokenizer,
  LlamaForCausalLM,
  LlamaTokenizer,
} from "@huggingface/transformers";
import { Jsonformer } from "../src/index.js";

async function main() {
  const spinner = ora({ text: "Initializing model: 0%" }).start();

  const model = await LlamaForCausalLM.from_pretrained(
    "onnx-community/Llama-3.2-1B-Instruct",
  );

  const tokenizer = await LlamaTokenizer.from_pretrained(
    "onnx-community/Llama-3.2-1B-Instruct",
  );

  const schema = {
    type: "object",
    properties: {
      name: { type: "string" },
      age: { type: "number" },
      is_student: { type: "boolean" },
      courses: {
        type: "array",
        items: { type: "string" },
      },
    },
  };

  const prompt =
    "Generate a person's information, including at least 5 courses, based on the following schema:";

  spinner.text = "Creating Jsonformer Instance...";
  const jsonformer = new Jsonformer(model, tokenizer, schema, prompt, {
    debug: true,
  });

  spinner.text = "Generating...";
  const result = await jsonformer.generate();

  spinner.succeed("Result successfully generated:");
  console.log(JSON.stringify(result, null, 2));
}

main().catch(console.error);
