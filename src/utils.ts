//@ts-nocheck
export const Callable = /** @type {any} */ class {
  constructor() {
    let closure = function (...args) {
      return closure._call(...args);
    };
    return Object.setPrototypeOf(closure, new.target.prototype);
  }

  _call(...args) {
    throw Error("Must implement _call method in subclass");
  }
};
