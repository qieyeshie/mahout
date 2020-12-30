package com.digquant;

import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClassifyIris {

    public static HashMap<String, List<Vector>> loadData() throws IOException {
        HashMap<String, List<Vector>> allData = new HashMap<>();
        String path = "F:\\mahout\\day3\\day3\\data\\Iris.csv";
        File file = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        // 1.1 去掉第一行标题
        String line = reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] lineArr = line.split(",");
            // 用x,y
            Vector vector = new DenseVector(4);
            double sepalLength = Double.parseDouble(lineArr[0]);
            double sepalWidth = Double.parseDouble(lineArr[1]);
            double petalLength = Double.parseDouble(lineArr[2]);
            double petalWidth = Double.parseDouble(lineArr[3]);
            vector.set(0, sepalLength);
            vector.set(1, sepalWidth);
            vector.set(2, petalLength);
            vector.set(3, petalWidth);
            String species = lineArr[4];
            List<Vector> vectors = allData.get(species);
            if (vectors == null) {
                //
                vectors = new ArrayList<>();
                allData.put(species, vectors);
            }
            vectors.add(vector);
        }
        return allData;
    }

    public static void main(String[] args) throws IOException {
        // 1 加载数据
        HashMap<String, List<Vector>> allData = loadData();
        // 1.1 使用 70%的数据用作训练，剩下的30%用来测试
        HashMap<String, List<Vector>> trainDataMap = new HashMap<>();
        HashMap<String, List<Vector>> testDataMap = new HashMap<>();
        for (Map.Entry<String, List<Vector>> entry : allData.entrySet()) {
            int trainNum = (int) (0.7f * entry.getValue().size());
            trainDataMap.put(entry.getKey(), entry.getValue().subList(0, trainNum));
            testDataMap.put(entry.getKey(), entry.getValue().subList(trainNum, entry.getValue().size()));
        }

        // 2.1 构建贝叶斯的数据格式
        //  因为有三个品种所以是三个分类，同时特征向量有4个
        double[][] datas = new double[3][4];

        // 2.2 把同一个品种的同一列数据相加
        int i = 0;
        for (Map.Entry<String, List<Vector>> entry : trainDataMap.entrySet()) {
            List<Vector> vectors = entry.getValue();
            for (int j = 0; j < vectors.size(); ++j) {
                Vector vector = vectors.get(j);
                for (int k = 0; k < vector.size(); ++k) {
                    datas[i][k] += vector.get(k);
                }
            }
            ++i;
        }
        // 生成权重矩阵
        DenseMatrix weightMatrix = new DenseMatrix(datas);
        // 2.3 把同一个品种所有的特征值相加, 构建特征总值
        int species1TotalFeature = 0;
        int species2TotalFeature = 0;
        int species3TotalFeature = 0;
        for (i = 0; i < 4; ++i) {
            species1TotalFeature += datas[0][i];
            species2TotalFeature += datas[1][i];
            species3TotalFeature += datas[2][i];
        }
        //
        DenseVector featureSum = new DenseVector(3);
        featureSum.set(0, species1TotalFeature);
        featureSum.set(1, species2TotalFeature);
        featureSum.set(2, species3TotalFeature);
        // 2.4 构建同一个特征值在所有品种的总和
        DenseVector labelSum = new DenseVector(6);
        for (i = 0; i < 4; ++i) {
            labelSum.set(i, datas[0][i] + datas[1][i] + datas[2][i]);
        }

        NaiveBayesModel naiveBayesModel = new NaiveBayesModel(weightMatrix, labelSum, featureSum, null, 1f, true);

        StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(naiveBayesModel);
        // 3 测试数据
        int correctNum = 0;
        int totalTestNum = 0; //
        i = 0;
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



}
