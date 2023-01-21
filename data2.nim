import nimib
nbInit

nbText: "by default numpy export has no header and separator is space"
nbCode:
  import datamancer

  let df = readCsv("data.csv", sep = ' ', colNames = @["x", "y"])

  echo df

nbCode:
  import ggplotnim

  ggplot(df, aes("x", "y")) + geom_point() + ggsave("data.png")

nbImage("data.png")

nbSave
# failing with both pixie and tikz backend (without me ever telling them to change backend)
