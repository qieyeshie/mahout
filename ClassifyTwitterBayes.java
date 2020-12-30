package com.digquant;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.Vectors;
import org.apache.mahout.vectorizer.TFIDF;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;
import org.junit.Test;

import java.io.*;
import java.util.*;

public class ClassifyTwitterBayes {


    public static void main(String[]args) throws IOException {
        // 1, 加载数据，构建特征向量值
        String path = "F:\\mahout\\day3\\data\\tweets-train.tsv";
        File file = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        Analyzer analyzer = new StandardAnalyzer();
        HashMap<String, List<Multiset<String>>> tokenMap = new HashMap<>();
        HashMap<String, Integer> wordDict = new HashMap<>();
        int wordIndex = 0;
        String line = null;
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split("\t");
            if (tokens.length < 3) {
                continue;
            }
            String label = tokens[0];
            String tweet = tokens[2];
            List<Multiset<String>> wordList = tokenMap.get(label);
            if (wordList == null) {
                wordList = new ArrayList<>();
                tokenMap.put(label, wordList);
            }
            Multiset<String> wordSet = ConcurrentHashMultiset.create();
            TokenStream ts = analyzer.tokenStream("text", new StringReader(tweet));
            CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
            ts.reset();
            while (ts.incrementToken()) {
                if (termAtt.length() > 0) {
                    String word = ts.getAttribute(CharTermAttribute.class).toString();
                    wordSet.add(word);
                    if (!wordDict.containsKey(word)) {
                        wordDict.put(word, wordIndex++);
                    }
                }
            }
            wordList.add(wordSet);
            ts.close();
        }
        // 构建字典
        HashMap<String, List<Vector>> vectorMap = new HashMap<>();
        for (Map.Entry<String, List<Multiset<String>>> entry : tokenMap.entrySet()) {
            List<Multiset<String>> wordList = entry.getValue();
            List<Vector> vectors = vectorMap.get(entry.getKey());
            if (vectors == null) {
                vectors = new ArrayList<>();
                vectorMap.put(entry.getKey(), vectors);
            }
            for (Multiset<String> wordSet : wordList) {
                Vector vector = new RandomAccessSparseVector(wordDict.size());
                for (String word : wordSet) {
                    int curWordIndex = wordDict.get(word);
                    double tfidf = calcTFIDF(word, wordSet, tokenMap);
                    vector.set(curWordIndex, tfidf);
                }
                vectors.add(vector);
            }
        }

        // 1.1 使用 70%的数据用作训练，剩下的30%用来测试
        HashMap<String, List<Vector>> trainDataMap = new HashMap<>();
        HashMap<String, List<Vector>> testDataMap = new HashMap<>();
        for (Map.Entry<String, List<Vector>> entry : vectorMap.entrySet()) {
            int trainNum = (int) (0.7f * entry.getValue().size());
            trainDataMap.put(entry.getKey(), entry.getValue().subList(0, trainNum));
            testDataMap.put(entry.getKey(), entry.getValue().subList(trainNum, entry.getValue().size()));
        }
        NaiveBayesModel naiveBayesModel = makeModel(trainDataMap, wordDict.size());
        StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(naiveBayesModel);
        // 3 测试数据
        int correctNum = 0;
        int totalTestNum = 0; //
        int i = 0;
        for (Map.Entry<String, List<Vector>> entry : testDataMap.entrySet()) {
            List<Vector> vectors = entry.getValue();
            for (int j = 0; j < vectors.size(); ++j) {
                Vector vector = vectors.get(j);
                Vector preV = classifier.classifyFull(vector);
                if (preV.maxValueIndex() == i) {
                    ++correctNum;
                }
            }
            totalTestNum += vectors.size();
            ++i;
        }
        System.out.printf("correctNum = %d correct precent = %f \n", correctNum, correctNum * 1.0 / totalTestNum);
    }

    static NaiveBayesModel makeModel(HashMap<String, List<Vector>> vectorMap, int featureNum) {
        double[][] datas = new double[vectorMap.size()][featureNum];
        // 2.2 把同一个品种的同一列数据相加
        int i = 0;
        for (Map.Entry<String, List<Vector>> entry : vectorMap.entrySet()) {
            List<Vector> vectors = entry.getValue();
            for (int j = 0; j < vectors.size(); ++j) {
                Vector vector = vectors.get(j);
                for (int k = 0; k < vector.size(); ++k) {
                    datas[i][k] += vector.get(k);
                }
            }
            ++i;
        }
        DenseMatrix weightMatrix = new DenseMatrix(datas);
        // 生成权重矩阵
        DenseVector featureSum = new DenseVector(vectorMap.size());
        for (i = 0; i < vectorMap.size(); ++i) {
            double totalNum = 0;
            for (int j = 0; j < featureNum; ++j) {
                totalNum += datas[i][j];
            }
            featureSum.set(i, totalNum);
        }

        // 2.4 构建同一个特征值在所有品种的总和
        Vector labelSum = new DenseVector(featureNum);
        for (i = 0; i < featureNum; ++i) {
            double totalNum = 0;
            for (int j = 0; j < vectorMap.size(); ++j) {
                totalNum += datas[j][i];
            }
            labelSum.set(i, totalNum);
        }

        return new NaiveBayesModel(weightMatrix, labelSum, featureSum, null, 1, true);
    }


    static double calcTFIDF(String word, Multiset<String> dstDoc, HashMap<String, List<Multiset<String>>> tokenMap) {
        double tf = dstDoc.count(word) * 1.0;
        double idf = 0;
        int containDoc = 0;
        int totalLineNum = 0;
        for (Map.Entry<String, List<Multiset<String>>> entry : tokenMap.entrySet()) {
            for (Multiset<String> wordSet : entry.getValue()) {
                if (wordSet.contains(word)) {
                    ++containDoc;
                }
            }
            totalLineNum += entry.getValue().size();
        }
        idf = Math.log(totalLineNum * 1.0 / (containDoc + 1));
        return tf * idf;

    }
}
