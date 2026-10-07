## Checks of the explanation's data: nim c -r docs/explain/test_explain.nim, and
## nim js -d:nodejs -r docs/explain/test_explain.nim for the JS side.
import std/[json, unittest]
import explain

suite "explain data":
  test "quantile as numpy's default":
    check quantile(@[3.0, 1.0, 2.0, 4.0], 0.5) == 2.5
    check quantile(@[3.0, 1.0, 2.0, 4.0], 0.25) == 1.75
    check quantile(@[5.0], 0.3) == 5.0

  test "toy25: 25 points, symmetric similarities, 0 on the diagonal":
    let input = toy25()
    check input.points.len == 25
    let s = input.similarityMatrix
    for i in 0 ..< 25:
      check s[i, i] == 0
      for k in 0 ..< 25: check s[i, k] == s[k, i]

  when not defined(js):
    let input = toy25()
    let params = paperParameters()
    let run = runAp(input, params)

    test "recording the messages does not change the result":
      var s = input.similarityMatrix
      let v = s.view
      v.prepare([run.preference], noise = false, seed = 0)
      let plain = v.affinityPropagation(params.maxits, params.convits, params.damping)
      check plain == run.result

    test "one record per iteration, consistent decisions":
      let its = run.iterations
      check its.r.len == run.result.iterations
      check its.a.len == its.r.len and its.exemplars.len == its.r.len
      check its.choice.len == its.r.len
      # the last exemplars are naffprop's before refinement: as many
      check its.exemplars[^1].len == run.result.exemplars.len
      # at convergence an exemplar chooses itself (not always before: an early
      # r(k, k) + a(k, k) > 0 need not be k's largest a(k, j) + r(k, j))
      for k in its.exemplars[^1]: check its.choice[^1][k] == k

    test "JSON round trip":
      var rounded = run
      rounded.roundMessages(4)
      let back = parseJson($(%rounded)).to(ApRun)
      check back == rounded

  when defined(js):
    test "the generated data parses in JS (run gen_data.nim first)":
      const data = staticRead("data/toy25.json")
      let run = parseJson(data).to(ApRun)
      check run.input.points.len == 25
      check run.iterations.r.len == run.result.iterations
      check run.iterations.r[0].n == 25

  test "roundSignificant":
    check roundSignificant(123.456, 4) == 123.5
    check roundSignificant(-0.00123456, 3) == -0.00123
    check roundSignificant(0.0, 4) == 0.0
