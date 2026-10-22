import test from "node:test";
import assert from "node:assert/strict";
import { parseValues } from "../app.js";

test("ranges preserve ordering, steps, negative values and deduplication", () => {
  assert.deepEqual(parseValues("-2..2:2, 5..1:2\n-2, January"), ["-2", "0", "2", "5", "3", "1", "January"]);
  assert.deepEqual(parseValues("1..3:0"), ["1", "2", "3"]);
  assert.deepEqual(parseValues("4..4"), ["4"]);
});

test("ranges are bounded before expansion", () => {
  assert.throws(() => parseValues("1..1000000000"), /limit 1000/);
  assert.throws(() => parseValues("1000000000..1"), /limit 1000/);
  assert.equal(parseValues("1..1000,1..1000").length, 1000);
  assert.throws(() => parseValues("1..1000, extra"), /limit 1000/);
  assert.throws(() => parseValues("1..600,601..1001"), /limit 1000/);
});

test("unsafe numeric ranges fail instead of stalling or losing precision", () => {
  assert.throws(() => parseValues("9007199254740992..9007199254740994"), /safe integers/);
  assert.throws(() => parseValues("1..3:9007199254740992"), /safe integers/);
  assert.deepEqual(parseValues("9007199254740990..9007199254740991"), ["9007199254740990", "9007199254740991"]);
});
