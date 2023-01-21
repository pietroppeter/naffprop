# dev notes

- wikipedia of algo: https://en.wikipedia.org/wiki/Affinity_propagation
- pdf of original science publication (personal use): https://warwick.ac.uk/fac/sci/dcs/research/combi/seminars/freydueck_affinitypropagation_science2007.pdf
- python implementation in sklearn: https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/cluster/_affinity_propagation.py#L37
- start with example data from python: https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html
- (had to install conda on my mac)
- link to Shiokawa's scalable version: https://scholar.google.it/scholar?hl=it&as_sdt=2005&sciodt=0%2C5&cites=17534306066151546043&scipsc=&q=Scalable+affinity+propagation+for+massive+datasets&btnG=
- let's start in 2d
- tabby not on nimble: https://github.com/treeform/tabby
  - tabby's docs: https://nimdocs.com/treeform/tabby/tabby.html
  - tabby does not work with tuple?
  - api in docs different from current api (parseContext has a name), seems not to support a float
- let's try again using ggplotnim/datamancer
  - reading without header uses colNames: https://scinim.github.io/Datamancer/datamancer.html#creating-a-dataframe-from-a-csv-slash-tsv-file
  - scatterplot https://scinim.github.io/getting-started/basics/basic_plotting.html
  - pixie backend by default? fails
  - if later I install ginger it uses tikz backend which also errors out
  - saved source in data1
  - see below for errors
- now converted data to json (thought of using p5nim but actually let's concentrate on algo)
- cran apcluster: https://cran.r-project.org/web/packages/apcluster/index.html
- where did I saw the trick of using a reduced version of similarity matrix? (only the triangular part)
- [x] implemented basic api for similarity matrix! :party_face:
- starting with a 2d point (generic later, then move points to its own module)
- [x] 8 test points
- [x] toSimMatrix
- [x] minimal matrix module

## similarity

- module with basic api to work with a similarity matrix (in a compact way)

## errors ggplotnim

```
❯ nim r data2 --nbShow
Hint: used config file '/Users/pietroppeter/nim/config/nim.cfg' [Conf]
Hint: used config file '/Users/pietroppeter/nim/config/config.nims' [Conf]
............................................................................................................................................................................................
/Users/pietroppeter/.nimble/pkgs/nimblas-0.2.2/nimblas/private/common.nim(50, 7) Hint: Using BLAS library with name: libblas.dylib [User]
................................................................................................................................................
/Users/pietroppeter/affprop/data.nim(12, 1) template/generic instantiation of `nbCode` from here
/Users/pietroppeter/nimib/src/nimib.nim(74, 18) template/generic instantiation of `captureStdout` from here
/Users/pietroppeter/.nimble/pkgs/ginger-0.3.13/ginger/backendPixie.nim(27, 29) Error: undeclared identifier: 'SolidPaint'
candidates (edit distance, scope distance); see '--spellSuggest': 
 (3, 4): 'SomePaint' [type declared in /Users/pietroppeter/pixie/src/pixie/paints.nim(34, 3)]
```

```
❯ nim r data2 --nbShow  
Hint: used config file '/Users/pietroppeter/nim/config/nim.cfg' [Conf]
Hint: used config file '/Users/pietroppeter/nim/config/config.nims' [Conf]
............................................................................................................................................................................................
/Users/pietroppeter/.nimble/pkgs/nimblas-0.2.2/nimblas/private/common.nim(50, 7) Hint: Using BLAS library with name: libblas.dylib [User]
............................................................................................................
/Users/pietroppeter/affprop/data.nim(12, 1) template/generic instantiation of `nbCode` from here
/Users/pietroppeter/nimib/src/nimib.nim(74, 18) template/generic instantiation of `captureStdout` from here
/Users/pietroppeter/.nimble/pkgs/ginger-0.4.1/ginger/backendTikZ.nim(11, 8) Error: cannot open file: latexdsl_nochecks
```