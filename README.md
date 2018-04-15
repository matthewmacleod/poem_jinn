# README

You have found the Poem Jinn:

https://en.wikipedia.org/wiki/Jinn

Welcome, let's generate some words!

Here's an example from text generated from a Shelley model:

```
At last from a marble bed-- 
Roses for a matron's head-- 
Violets by dark who die, and grove 
From the world's carnival. 
```

## Prepare data

Assumes project Gutenburg text files.

http://www.gutenberg.org/

```
cd data
python src/process_data.py data/author.txt
```

Where `author.txt` is the poet to learn patterns from.

This will generate a file `data/author_clean.txt`

## To run

If we had cleaned the file `shelley.txt`, we'd then run:

```
python src/main.py --target_text shelley_clean
```

## To sample

Generate predictions based on the model

```
python src/sample.py --target_text shelley_clean
```

## License

MIT, see LICENSE.md

This is an open source poem generator.


