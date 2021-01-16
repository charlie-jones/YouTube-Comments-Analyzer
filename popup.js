/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var entered = false;
var comments_loaded = false;
var yt_comments = []
chrome.tabs.query({currentWindow: true, active: true},
  function (tabs) {
    try {
      var video_id = tabs[0].url.split('v=')[1];
      var ampersandPosition = video_id.indexOf('&');
      if(ampersandPosition != -1) {
          video_id = video_id.substring(0, ampersandPosition);
      }
      // HTTP request -- [hidden code]
      Http.onreadystatechange = (e) => {
          if (!comments_loaded) {
              var data;
              var contin = true;
              try {
                data = JSON.parse(Http.responseText);
              } catch (err) {
                contin = false;
              }
              if (contin) {
                for (var i = 0; i < data.items.length; i++) {
                    yt_comments.push(data.items[i].snippet.topLevelComment.snippet.textOriginal);
                    var replyCount = data.items[i].snippet.totalReplyCount;
                    if (replyCount < 1) continue;
                    try {
                        replyCount = data.items[i].replies.comments.length;
                        yt_comments.push(data.items[i].replies.comments[0].snippet.textOriginal);
                    } catch (err) {}
                }
                comments_loaded = true;
                addCommentsAndRun(yt_comments);
              }
          }
      }
    } catch (err) {
      document.getElementById('reload-message').style.display = 'block';
      document.getElementById('initial content').style.display = 'none';
    }
  });
var model;
var metadataJson;
var sentimentMetadata;
var indexFrom;
var maxLen;
var wordIndex;
var vocabularySize;
async function loadEverything() {
  model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json');
  metadataJson = await fetch('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json');
  sentimentMetadata = await metadataJson.json();
  indexFrom = sentimentMetadata['index_from'];
  maxLen = sentimentMetadata['max_len'];
  wordIndex = sentimentMetadata['word_index'];
  vocabularySize = sentimentMetadata['vocabulary_size'];
}
async function padSequences(
  sequences, padding = 'pre', truncating = 'pre', value = 0) {
// TODO(cais): This perhaps should be refined and moved into tfjs-preproc.
return sequences.map(seq => {
  // Perform truncation.
  if (seq.length > maxLen) {
    if (truncating === 'pre') {
      seq.splice(0, seq.length - maxLen);
    } else {
      seq.splice(maxLen, seq.length - maxLen);
    }
  }

  // Perform padding.
  if (seq.length < maxLen) {
    const pad = [];
    for (let i = 0; i < maxLen - seq.length; ++i) {
      pad.push(value);
    }
    if (padding === 'pre') {
      seq = pad.concat(seq);
    } else {
      seq = seq.concat(pad);
    }
  }

  return seq;
});
}
async function predict(text) {
  // Convert to lower case and remove all punctuations.
  const inputText =
      text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  // Convert the words to a sequence of word indices.
  const sequence = inputText.map(word => {
    var wdIndex = wordIndex[word] + indexFrom;
    if (wdIndex > vocabularySize) {
      wdIndex = 2;
    }
    return wdIndex;
  });
  // Perform truncation and padding.
  const paddedSequence = await padSequences([sequence]);
  const input = tf.tensor2d(paddedSequence, [1, maxLen]);
  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  return score;
}
var sumOfScores = 0;
var countScores = 0;
var comments = [];
var comment_scores = [];
function show3comments() {
  if (comments.length >= 3) {
    var divComments = document.getElementById('comments');
    for (var i = 0; i < 3; i++) {
      var cur_comment = comments[i];
      divComments.innerHTML += "<div style='overflow-wrap: break-word; font-size: 14px'>" + cur_comment + ": " + "<b>"+ comment_scores[i].toFixed(3) + "</b>" + "</div>"
    }
  }
  document.getElementById('show-comments').removeEventListener('click', show3comments);
  document.getElementById('show-comments').disabled = true;
}
async function computeScores() {
  await loadEverything();
  for (message of comments) {
    await predict(message).then((score) => {
      comment_scores.push(score);
      sumOfScores += score;
      countScores++;
    });
  }
  var mean_score = (sumOfScores / countScores).toFixed(3);
  entered = true;
  document.getElementById('reload-message').style.display = 'none';
  document.getElementById('initial content').style.display = 'none';
  document.getElementById('sentiment-analysis').style.display = 'block';
  document.getElementById('sentiment-score').innerHTML = mean_score;
  document.getElementById('score-slider').value = mean_score*1000;
  document.getElementById('score-2').style.marginLeft = Math.floor(mean_score*175+15).toString() + "px";
  document.getElementById('score-2').innerHTML = mean_score;
  document.getElementById('comments-analyzed').innerHTML = comments.length;
  document.getElementById('show-comments').style.marginBottom = '20px';
  document.getElementById('show-comments').addEventListener('click', show3comments);
}
function addCommentsAndRun(youtube_comments) {
  for (comment of youtube_comments) {
    comments.push(comment);
  }
  computeScores();
}
