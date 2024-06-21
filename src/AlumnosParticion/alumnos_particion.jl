using CSV, DataFrames
using Random, StatsBase

dataset = DataFrame(CSV.File("alumnos.txt"))
mitad = trunc(Int, nrow(dataset)/2)

Random.seed!( 96005236357 )
naranja = sample(dataset[ :, :id],  mitad, replace =false )

dataset[ [ x in naranja for x in dataset[:id]] , :]
