import chalk from "chalk";

function recursivePrint(
  obj: any,
  indent: number = 0,
  isLastElement: boolean = true,
): void {
  if (obj === null) {
    process.stdout.write(chalk.green("null"));
    process.stdout.write(isLastElement ? "\n" : ",\n");
    return;
  }

  if (typeof obj === "object") {
    if (Array.isArray(obj)) {
      process.stdout.write("[\n");
      obj.forEach((value, index) => {
        process.stdout.write(" ".repeat(indent + 2));
        recursivePrint(value, indent + 2, index === obj.length - 1);
      });
      process.stdout.write(
        `${" ".repeat(indent)}]${isLastElement ? "\n" : ",\n"}`,
      );
    } else {
      process.stdout.write("{\n");
      const keys = Object.keys(obj);
      keys.forEach((key, index) => {
        process.stdout.write(`${" ".repeat(indent + 2)}${key}: `);
        recursivePrint(obj[key], indent + 2, index === keys.length - 1);
      });
      process.stdout.write(
        `${" ".repeat(indent)}}${isLastElement ? "\n" : ",\n"}`,
      );
    }
  } else {
    let output = typeof obj === "string" ? `"${obj}"` : String(obj);
    process.stdout.write(chalk.green(output));
    process.stdout.write(isLastElement ? "\n" : ",\n");
  }
}

export function highlightValues(value: any): void {
  recursivePrint(value);
}
