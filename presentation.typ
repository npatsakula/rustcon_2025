#import "@preview/polylux:0.4.0": *

#set page(paper: "presentation-16-9")
#set text(font: "Liberation Sans", size: 20pt)

#let rustconf-theme(body) = {
  set page(
    fill: rgb("#1a1a1a"),
    margin: (x: 2em, y: 2em),
  )
  set text(fill: rgb("#f0f0f0"))

  show heading.where(level: 1): it => {
    set text(size: 32pt, weight: "bold", fill: rgb("#ff6b35"))
    it
  }

  show heading.where(level: 2): it => {
    set text(size: 24pt, weight: "bold", fill: rgb("#f7931e"))
    it
  }

  show raw.where(block: true): it => {
    set text(size: 14pt, font: "Fira Code")
    block(
      fill: rgb("#2d2d2d"),
      inset: 1em,
      radius: 0.5em,
      width: 100%,
      it
    )
  }

  show raw.where(block: false): it => {
    box(
      fill: rgb("#2d2d2d"),
      inset: (x: 0.3em, y: 0.1em),
      radius: 0.2em,
      it
    )
  }

  body
}

#show: rustconf-theme

// Title Slide
#slide[
  #align(center + horizon)[
    #text(size: 48pt, weight: "bold", fill: rgb("#ff6b35"))[
      Morok: A High-Performance ML Framework in Rust
    ]

    #v(2em)

    #text(size: 24pt)[
      RustConf 2025
    ]
  ]
]

// Problems Slide
#slide[
  = Проблемы крупных ML фреймворков

  #v(1em)

  #set text(size: 24pt)

  #enum(
    numbering: "1.",
    [Потребность работать в зоопарке аппаратного ускорения],
    [Сложно вносить изменения и профилировать],
    [Сложная и дорогая DevOps/MLOps инфраструктура],
    [Небольшая площать интеропа],
    [Невозможность использовать большую часть экосистемы]
  )
]

// Frameworks Comparison - Overview
#slide[
  = ML фреймворки: история

  #v(2em)

  #align(center)[
    #image("frameworks_histogram.svg", width: 95%)
  ]
]

// Tinygrad Capabilities
#slide[
  = Tinygrad

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      == Аппаратные ускорители

      #list(
        [CPU],
        [CUDA],
        [AMD],
        [METAL],
        [WebGPU],
        [OpenCL],
        [Qualcomm]
      )
    ],
    [
      == Особенности

      #enum(
        [Гибкий фроентенд a la PyTorch],
        [Ленивая семантика a la MLX],
        [JIT компиляция],
        [Полу-автоматическое шардирование],
        [Богатые средства интроспекции],
        [Легко добавить новый таргет],
      )
    ]
  )
]

// Tinygrad Architecture
#slide[
  = Архитектура Tinygrad

  #v(1em)

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    align: center,
    [
      *Tensor Layer*\
      PyTorch-like API\
      Lazy evaluation\
      #sym.arrow.b\
      UOp graph
    ],
    [
      *Rangeify*\
      Eliminate movement\
      (reshape/permute)\
      #sym.arrow.b\
      Index expressions\
      \+ RANGE nodes
    ],
    [
      *Schedule*\
      Kernel extraction\
      Dependency analysis\
      Fusion\
      #sym.arrow.b\
      ScheduleItem list
    ],
    [
      *Codegen*\
      Lower UOp\
      Simplify\
      #sym.arrow.b\
      C/CUDA code
    ],                                                                                                                             [
      *Runtime*\                                                                                                                     Compile (clang/nvcc)\
      Buffer allocation\                                                                                                             #sym.arrow.b\
      Execute kernels                                                                                                              ]
  )
]

// Frontend Example - Python Code
#slide[
  = Frontend: PyTorch-like Code

  ```python
  a = Tensor([1.0, 2.0, 3.0])
  b = Tensor([4.0, 5.0, 6.0])
  c = Tensor([2.0, 2.0, 2.0])

  # Perform operations (lazy - no computation yet!)
  result = (a + b) * c
  ```
]

// Frontend Example - IR Tree
#slide[
  = Frontend: Generated IR Tree

  #v(0.5em)

  #text(size: 12pt, font: "Fira Code")[
    ```
    MUL (dtype=dtypes.float)                    ← Final: (a+b)*c
      └─ input[0]:
        ADD (dtype=dtypes.float)                ← a + b
          └─ input[0]:
            COPY (dtype=dtypes.float)           ← Copy 'a'
              └─ input[0]:
                BUFFER (dtype=dtypes.float)
                  └─ input[0]: UNIQUE (dtype=dtypes.void)
                  └─ input[1]: DEVICE (dtype=dtypes.void)
          └─ input[1]:
            COPY (dtype=dtypes.float)           ← Copy 'b'
              └─ input[0]:
                BUFFER (dtype=dtypes.float)
                  └─ input[0]: UNIQUE (dtype=dtypes.void)
                  └─ input[1]: DEVICE (dtype=dtypes.void)
      └─ input[1]:
        COPY (dtype=dtypes.float)               ← Copy 'c'
          └─ input[0]:
            BUFFER (dtype=dtypes.float)
              └─ input[0]: UNIQUE (dtype=dtypes.void)

    Total nodes: 13
    ```
  ]
]

// Codegen Example
#slide[
  = Codegen: Generated Kernel

  ```c
  __kernel void E_3(
      global float* data0,           // Output: result
      const global float* data1,     // Input: a
      const global float* data2,     // Input: b
      const global float* data3)     // Input: c
  {
      int gidx0 = get_global_id(0);  // Global thread ID

      if (gidx0 < 3) {                // Bounds check
          float val0 = data1[gidx0];  // Load a[i]
          float val1 = data2[gidx0];  // Load b[i]
          float val2 = data3[gidx0];  // Load c[i]

          float acc0 = (val0 + val1);  // a + b
          float acc1 = (acc0 * val2);  // (a+b) * c

          data0[gidx0] = acc1;         // Store result
      }
  }
  ```
]

// Tinygrad Problems
#slide[
  = Tinygrad's Problems

  #v(1em)

  #text(size: 24pt, fill: rgb("#ff6b35"))[
    Python #sym.arrow.r weak IR, poor performance
  ]

  #v(2em)

  #enum(
    [The lack of proper sum types and strict type checking leads to non-exhaustive tree pattern matching],
    [On large graphs, the time taken by optimizer passes becomes significant]
  )
]

// Liho Slide (placeholder)
#slide[
  = Liho

  #v(2em)

  #align(center)[
    #text(size: 24pt, style: "italic", fill: rgb("#888888"))[
      (Details about Liho to be added)
    ]
  ]
]

// Morok Architecture
#slide[
  = Morok Architecture

  #v(2em)

  #enum(
    [Fully typed IR],
    [Four stage compiler],
    [AOT mode]
  )
]

// Morok Performance
#slide[
  = Morok Performance

  #v(2em)

  #text(size: 24pt)[
    Benchmarks:
  ]

  #v(1em)

  #enum(
    [MNIST digit classification],
    [LLAMA 3 token generation]
  )

  #v(1em)

  #align(center)[
    #text(style: "italic", fill: rgb("#888888"))[
      (Performance graphs to be added)
    ]
  ]
]

// Thunderkittens
#slide[
  = Thunderkittens

  #v(2em)

  #align(center)[
    #text(size: 28pt)[
      High-performance kernel DSL
    ]
  ]
]

// Future Plans
#slide[
  = Future Plans

  #v(2em)

  #enum(
    [Implement `morok-kittens`, port FA 2/3/4],
    [Implement LLVM generation for all platforms],
    [Implement more models]
  )
]

// Thank You Slide
#slide[
  #align(center + horizon)[
    #text(size: 48pt, weight: "bold", fill: rgb("#ff6b35"))[
      Thank You!
    ]

    #v(2em)

    #text(size: 24pt)[
      Questions?
    ]
  ]
]
