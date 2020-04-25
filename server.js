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

app.route('/:coglangSentence').get(function(req, res, next) {

  // get request parameter data
  let coglangSentence = req.params.coglangSentence;
  console.log('some kind of custom request', coglangSentence);
  coglangSentence = coglangSentence.toLowerCase();
  coglangSentence = coglangSentence.replace(/  +/g,' ').trim();
  
  // split into words
  coglangSentence = coglangSentence.split(' ');
  
  // get just the ones marked as missing
  const missingWords = coglangSentence.filter((word) => {
    return word.startsWith('[') && word.endsWith(']');
  }).map((word) => {
    return word.replace('[', '').replace(']', '');
  });

  // set up response data
  const outputData = {
    missingWord: [missingWords[0]],
    suggestions: []
  };

  // get dictionary before translate
  fs.readFile('embeddings.txt', 'utf8', async function (err,data) {
    if (err) {
      return console.log(err);
    }
    
    const noMissingWords = missingWords.length === 0;
    if (noMissingWords) {
      res.type('json').send(outputData);
      return; // exit early
    }
    
    // // TODO: go through each embedding
    // for (let embedding of data) {
    // }
    
    // TODO: just get closest match to first missing word
    
    // for now, just check closeness of match of first missing word with one word in the vocab
    const similarityPercent = await useModel('hello', missingWords[0]);
    console.log(8, similarityPercent)
    
    outputData.suggestions = similarityPercent || 0;
    
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

async function useModel(sentence1, sentence2) {
  console.log(1, sentence1, sentence2);
  // uses Universal Sentence Encoder (U.S.E.):
  return await use.load().then(async (model) => {
    console.log(2, 'inside use.load');
    const similarityFraction = await embedSentences(model, sentence1, sentence2);
    console.log(7, 'similarityFraction ' + similarityFraction)
    return Math.round(similarityFraction * 100 * 100) / 100 + '%';
  });
}

async function embedSentences(model, sentence1, sentence2) {
  const sentences = [sentence1, sentence2];
  return await model.embed(sentences).then(async (embeddings) => {
    console.log(3, 'inside model.embed');
    const embeds = await embeddings.arraySync();
    console.log(4, 'did arraySync after got embeddings');
    const sentence1Embedding = embeds[0];
    const sentence2Embedding = embeds[1];
    const similarityPercent = await getSimilarityPercent(sentence1Embedding, sentence2Embedding);
    console.log(6, 'similarityPercent ' + similarityPercent);
    return similarityPercent;
  });
}

async function getSimilarityPercent(embed1, embed2) {
  const similarity = await cosineSimilarity(embed1, embed2);
  console.log(5, 'got similarity ' + similarity);
  // cosine similarity -> % when doing text comparison, since cannot have -ve term frequencies: https://en.wikipedia.org/wiki/Cosine_similarity
  return similarity;
}

async function cosineSimilarity(a, b) {
  // https://towardsdatascience.com/how-to-build-a-textual-similarity-analysis-web-app-aa3139d4fb71

  const magnitudeA = await Math.sqrt(dotProduct(a, a));
  const magnitudeB = await Math.sqrt(dotProduct(b, b));
  if (magnitudeA && magnitudeB) {
    // https://towardsdatascience.com/how-to-measure-distances-in-machine-learning-13a396aa34ce
    return await dotProduct(a, b) / (magnitudeA * magnitudeB);
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
