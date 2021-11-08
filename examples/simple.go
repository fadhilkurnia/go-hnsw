package main

import (
	"fmt"
	"math/rand"
	"time"

	hnsw "github.com/Bithack/go-hnsw"
)

func main() {

	const (
		M              = 32
		efConstruction = 400
		efSearch       = 100
		K              = 10
	)

	var zero hnsw.Point = make([]float32, 128)

	h := hnsw.New(M, efConstruction, zero)
	h.Grow(10000)

	fmt.Printf(">>> Generating random data points ...\n")
	for i := 1; i <= 10000; i++ {
		h.Add(randomPoint(), uint32(i))
		//if (i)%1000 == 0 {
		//	fmt.Printf("%v points added\n", i)
		//}
	}

	fmt.Printf(">>> Generating queries and calculating true answers using bruteforce search ...\n")
	queries := make([]hnsw.Point, 1000)
	truth := make([][]uint32, 1000)
	for i := range queries {
		queries[i] = randomPoint()
		result := h.SearchBrute(queries[i], K)
		truth[i] = make([]uint32, K)
		for j := K - 1; j >= 0; j-- {
			item := result.Pop()
			truth[i][j] = item.ID
		}
	}

	fmt.Printf(">>> Now searching with HNSW ...\n")
	hits := 0
	start := time.Now()
	for i := 0; i < 1000; i++ {
		result := h.Search(queries[i], efSearch, K)
		for j := 0; j < K; j++ {
			item := result.Pop()
			for k := 0; k < K; k++ {
				if item.ID == truth[i][k] {
					hits++
				}
			}
		}
	}
	stop := time.Since(start)

	fmt.Printf("  %v queries/second (single thread)\n", 1000.0/stop.Seconds())
	fmt.Printf("  Average search time per query: %vms (efSearch=%d, K=%d)\n", float64(stop.Milliseconds())/1000.0, efSearch, K)
	fmt.Printf("  Average 10-NN precision: %v\n", float64(hits)/(1000.0*float64(K)))
	fmt.Println(h.Stats())

}

func randomPoint() hnsw.Point {
	var v hnsw.Point = make([]float32, 128)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
