'use strict';

const fs = require('fs');
// require("@tensorflow/tfjs-node");
const use = require("@tensorflow-models/universal-sentence-encoder");
const express = require('express');
const app = express();

if (!process.env.DISABLE_XORIGIN) {
  app.use(function(req, res, next) {
    var allowedOrigins = ['https://narrow-plane.gomix.me', 'https://www.freecodecamp.com'];
    var origin = req.headers.origin || '*';
    if(!process.env.XORIG_RESTRICT || allowedOrigins.indexOf(origin) > -1){
         console.log(origin);
         res.setHeader('Access-Control-Allow-Origin', origin);
         res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    }
    next();
  });
}

app.use('/public', express.static(process.cwd() + '/public'));

app.route('/:unknownWord').get(function(req, res, next) {

  // get request parameter data
  let unknownWord = req.params.unknownWord;
  console.log('some kind of custom unknown word', unknownWord);
  unknownWord = unknownWord.toLowerCase();
  unknownWord = unknownWord.replace(/ /g, '').replace(/\[/g, '').replace(/\]/g, '');

  // set up response data
  const outputData = {
    missingWord: unknownWord,
    suggestions: []
  };

  // get dictionary before translate
  fs.readFile('embeddings.txt', 'utf8', async function (err,data) {
    if (err) {
      return console.log(err);
    }
    
    if (!unknownWord) {
      res.type('json').send(outputData);
      return; // exit early
    }
    
    // just get closest 5 matches to the unknown word:
    const mostSimilarWords = await useModel(unknownWord);
    
    outputData.suggestions = mostSimilarWords;
    
    console.log(outputData);

    // finally return JSON response
    res.type('json').send(outputData);

  });
});
  

app.route('/').get(function(req, res) {
  res.sendFile(process.cwd() + '/views/index.html');
})


// Respond not found to all the wrong routes
app.use(function(req, res, next){
  res.status(404);
  res.type('txt').send('404: Not found');
});


// Error Middleware
app.use(function(err, req, res, next) {
  if(err) {
    res.status(err.status || 500)
      .type('txt')
      .send(err.message || 'SERVER ERROR');
  }  
})


app.listen(process.env.PORT, function () {
  console.log('Node.js listening ...');
});



// custom function(s):

function createDictionary(str) { // creates a "hash table" for faster searching
  var ht = {};
  var l = str.split('\n');
  for (var i in l) {
    var entry = l[i].split(',');
    var eng = entry[1];
    var cog = entry[0];
    var typ = entry[entry.length-1];
    ht[eng] = {'cog':cog,'type':typ};
  }
  return ht;
}

function getShortForm(cog) {
  var vowels = 'aeiou';
  var indexStopBefore = cog.length;
  var vowelCount = 0;
  for (var i in cog) {
    var letter = cog[i];
    if (vowels.includes(letter)) {
      vowelCount += 1;
      if (vowelCount >= 2) {
        // index2ndLastVowel = parseInt(i)+1; break;
        if (cog[parseInt(i)+1] !== null) {
          if (vowels.includes(cog[parseInt(i)+1])) {
            indexStopBefore = parseInt(i)+1;
          } else {
            indexStopBefore = parseInt(i)+2;
          }
          break;
        }
      }
    }
  }
  return cog.slice(0,indexStopBefore);
}



// Tensorflow.js stuff:

async function useModel(inputWord) {
  // uses Universal Sentence Encoder (U.S.E.):
  const mostSimilarWords = await use.load().then(async (model) => {
    return await getEmbedding(model, inputWord);
  });
  return mostSimilarWords;
}

function getEmbedding(model, inputWord) {
  return model
    .embed([inputWord])
    .then((inputEmbeddings) => {
      const embeds = inputEmbeddings.arraySync();
      const wordEmbedding = embeds[0];

      const embedsData = readFile("embeddings.txt");
      const lines = embedsData.split("\n");
      if (lines.length === 0) return []; // exit if no data

      const similarities = getAllSimilarityScores(wordEmbedding, embedsData);
      const top5 = getTop5Similarities(similarities);

      const englishData = readFile("out_english.txt");
      const mostSimilarWords = getMostSimilarWordsFromFile(top5, englishData);
      // console.log(mostSimilarWords);
      return mostSimilarWords;
    })
    .catch((err) => {
      console.log(err);
      return [];
    });
}

function readFile(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function getAllSimilarityScores(wordEmbedding, data) {
  const lines = data.split("\n");
  if (lines.length === 0) return []; // exit if no data
  const similarities = []; // TODO: use a max heap instead? Reference: https://github.com/hchiam/learning-google-closure-library/blob/master/goog-closure-example.js
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    const referenceEmbedding = line.split(",").map((n) => Number(n));
    if (referenceEmbedding.length === 0) continue; // skip empty line
    const similarity = getSimilarityPercent(wordEmbedding, referenceEmbedding);
    similarities.push({ similarity, index });
  }
  // console.log("similarities.length: " + similarities.length);
  return similarities;
}

function getSimilarityPercent(wordEmbedding, referenceEmbedding) {
  const similarity = cosineSimilarity(wordEmbedding, referenceEmbedding);
  // cosine similarity -> % when doing text comparison, since cannot have -ve term frequencies: https://en.wikipedia.org/wiki/Cosine_similarity
  return similarity;
}

function cosineSimilarity(a, b) {
  // https://towardsdatascience.com/how-to-build-a-textual-similarity-analysis-web-app-aa3139d4fb71

  const magnitudeA = Math.sqrt(dotProduct(a, a));
  const magnitudeB = Math.sqrt(dotProduct(b, b));
  if (magnitudeA && magnitudeB) {
    // https://towardsdatascience.com/how-to-measure-distances-in-machine-learning-13a396aa34ce
    return dotProduct(a, b) / (magnitudeA * magnitudeB);
  } else {
    return 0;
  }
}

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function getTop5Similarities(similarities) {
  return similarities
    .sort(function descending(a, b) {
      return b.similarity - a.similarity;
    })
    .slice(0, 5);
  // console.log("top 5 similar indices: " + top5.map((x) => x.index));
  // console.log("top 5 similarities: " + top5.map((x) => x.similarity));
}

function getMostSimilarWordsFromFile(top5, data) {
  const mostSimilarWords = [];
  const lines = data.split("\n");
  top5.forEach((similarity) => {
    const index = similarity.index;
    const word = lines[index];
    mostSimilarWords.push(word);
  });
  // console.log("most similar words: " + mostSimilarWords);
  return mostSimilarWords;
}
