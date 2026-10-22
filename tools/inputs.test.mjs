import test from "node:test";
import assert from "node:assert/strict";
import { parseValues, toTemplate } from "../app.js";

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

test("template cleanup preserves leading years and negative values", () => {
  assert.equal(toTemplate("1999 was the year everything changed.", "1999", ["1998", "1999", "2000"]), "{X} was the year everything changed.");
  assert.equal(toTemplate("-5 was the number on the door.", "-5", ["-5", "0", "5"]), "{X} was the number on the door.");
});

test("template cleanup removes list markers, bold formatting and quotes", () => {
  for (const prefix of ["1. ", "2) ", "- ", "* ", "• ", "  "]) {
    assert.equal(toTemplate(`${prefix}1999 was the year everything changed.`, "1999", ["1998", "1999", "2000"]), "{X} was the year everything changed.");
  }
  assert.equal(toTemplate('**1999** was the year everything changed.', "1999", ["1999"]), "{X} was the year everything changed.");
  assert.equal(toTemplate('"1999 was the year everything changed."', "1999", ["1999"]), "{X} was the year everything changed.");
});

test("template validation still rejects repeated anchors and other numeric values", () => {
  assert.equal(toTemplate("1999 was repeated as 1999 in the report.", "1999", ["1999"]), null);
  assert.equal(toTemplate("1999 was compared with 2000 in the report.", "1999", ["1999", "2000"]), null);
  assert.equal(toTemplate("1999 was listed on page 42 of the report.", "1999", ["1999"]), null);
});
