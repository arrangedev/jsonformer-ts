# jsonformer-ts 
A Typescript port of [Jsonformer](https://github.com/1rgs/jsonformer).

## Overview
This is a port of the Jsonformer library originally written in Python. This repo aims to replicate it as close as possible, allowing Typescript developers to take advantage of high-quality structured JSON outputs from language models. 

Like the original, the following schema types are supported:
- number
- string
- boolean
- array
- object

## Installation
NPM:
```bash
npm i @arrangedev/jsonformer-ts
```

yarn:
```bash
yarn add @arrangedev/jsonformer-ts
```

## Example
Here's a basic example of `jsonformer-ts` in action:
```Typescript
async function main() {
  console.log("Loading model and tokenizer...");
  const model = await LlamaForCausalLM.from_pretrained(
    "Xenova/TinyLlama-1.1B-Chat-v1.0",
    { model_file_name: "model" },
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

  console.log("Creating Jsonformer instance...");
  const jsonformer = new Jsonformer(model, tokenizer, schema, prompt, {
    debug: true,
  });

  console.log("Generating data...");
  const result = await jsonformer.generate();

  console.log("\nGenerated result:");
  console.log(JSON.stringify(result, null, 2));
}
```

Run the example with:
```bash
yarn example:basic
```

## License
This project is MIT licensed, like the original. See [here](https://github.com/1rgs/jsonformer?tab=MIT-1-ov-file#readme) for more information.
