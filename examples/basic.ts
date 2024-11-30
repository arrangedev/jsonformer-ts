import ora from "ora";
import { LlamaForCausalLM, LlamaTokenizer } from "@xenova/transformers";
import { Jsonformer } from "../src/index.js";

async function main() {
  const spinner = ora({ text: "Initializing model: 0%" }).start();

  const model = await LlamaForCausalLM.from_pretrained(
    "Xenova/TinyLlama-1.1B-Chat-v1.0",
    {
      progress_callback: (progress: any) => {
        if (!progress.progress) return;
        spinner.text = `Initializing Model: ${progress.progress}%`;
      },
    },
  );

  const tokenizer = await LlamaTokenizer.from_pretrained(
    "Xenova/TinyLlama-1.1B-Chat-v1.0",
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

  spinner.text = "Creating Jsonformer Instance...";
  const jsonformer = new Jsonformer(model, tokenizer, schema, prompt, {
    debug: false,
  });

  spinner.text = "Generating...";
  const result = await jsonformer.generate();

  spinner.succeed("Result successfully generated:");
  console.log(JSON.stringify(result, null, 2));
}

main().catch(console.error);
