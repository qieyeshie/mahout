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

public class ClassifyTwitterSGD {

    static class VectorHolder {
        Vector v;
        int target;
        String targetName;
    }


    private static  HashMap<String, List<Multiset<String>>> tokenMap = new HashMap<>();


    public static List<VectorHolder> loadData() throws IOException {
        // 1, 加载数据，构建特征向量值
        String path = "F:\\mahout\\day3\\data\\tweets-train.tsv";
        File file = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        Analyzer analyzer = new StandardAnalyzer();

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
                }
            }
            wordList.add(wordSet);
            ts.close();
        }
        // 构建特征向量
        List<VectorHolder> allData = new ArrayList<>();
        int target = 0;
        FeatureVectorEncoder encoder = new StaticWordValueEncoder("body");
        FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");
        encoder.setProbes(2);
        for (Map.Entry<String, List<Multiset<String>>> entry : tokenMap.entrySet()) {
            List<Multiset<String>> wordList = entry.getValue();
            for (Multiset<String> wordSet : wordList) {
                VectorHolder vectorHolder = new VectorHolder();
                vectorHolder.target = target;
                vectorHolder.targetName = entry.getKey();
                vectorHolder.v = new RandomAccessSparseVector(10000);
                bias.addToVector("", 1, vectorHolder.v);
                for (String word : wordSet) {
                    encoder.addToVector(word, calcTFIDF(word, wordSet, tokenMap), vectorHolder.v);
                }
                allData.add(vectorHolder);
            }
            ++target;
        }
        return allData;
    }


    public static void main(String[] args) throws IOException {
        List<VectorHolder> allData = loadData();
        // 创建一个随机种子
        Random random = RandomUtils.getRandom();
        Collections.shuffle(allData, random);
        int trainNum = (int) (0.7 * allData.size());
        List<VectorHolder> train = allData.subList(0, trainNum);
        List<VectorHolder> test = allData.subList(trainNum, allData.size());

        OnlineLogisticRegression lr = new OnlineLogisticRegression(tokenMap.size(), 10000, new L1())
                .alpha(1) // 学习率， 指数下降
                .stepOffset(1000) // 衰减方式
                .decayExponent(0.5) // 衰减率
                .lambda(1.0e-5) // 正则化权重
                .learningRate(40); // 初始学习率;

        for (int pass = 0; pass < 100; pass++) {
            Collections.shuffle(train, random);
            for (VectorHolder vHolder : train) {
                lr.train(vHolder.target, vHolder.v);
            }
        }
        int correctNum = 0;

        for (int i = 0; i < test.size(); ++i) {
            VectorHolder vectorHolder = test.get(i);
            Vector preV = lr.classifyFull(vectorHolder.v);
            if (preV.maxValueIndex() == vectorHolder.target) {
                ++correctNum;
            }
        }
        System.out.printf("correctNum = %d correct precent = %f \n", correctNum, correctNum * 1.0 / test.size());
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
