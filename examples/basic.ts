import { LlamaForCausalLM, LlamaTokenizer } from "@xenova/transformers";
import { Jsonformer } from "../src/index.js";

async function main() {
  console.log("Loading model and tokenizer...");
  const model = await LlamaForCausalLM.from_pretrained(
    "onnx-community/Llama-3.2-1B-Instruct",
    { model_file_name: "model" },
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
    "Generate a person's information based on the following schema:";

  console.log("Creating Jsonformer instance...");
  const jsonformer = new Jsonformer(model, tokenizer, schema, prompt, {
    debug: true,
  });

  console.log("Generating data...");
  const result = await jsonformer.generate();

  console.log("\nGenerated result:");
  console.log(JSON.stringify(result, null, 2));
}

main().catch(console.error);
