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

public class PredictStock {

    public static HashMap<Integer, List<Vector>> loadData() throws IOException {
        // 1, 加载数据
        HashMap<Integer, List<Vector>> allData = new HashMap<>();
        String path = "F:\\mahout\\day3\\day3\\data\\stock_data.csv";
        File file = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        // 1.1 去掉第一行标题
        String line = reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] lineArr = line.split(",");
            // 用x,y
            Vector vector = new DenseVector(5);
            double open = Double.parseDouble(lineArr[1]);
            double high = Double.parseDouble(lineArr[2]);
            double low = Double.parseDouble(lineArr[3]);
            double volume = Double.parseDouble(lineArr[4]);
            double amount = Double.parseDouble(lineArr[5]);
            int up = Integer.parseInt(lineArr[7]);
            vector.set(0, open);
            vector.set(1, high);
            vector.set(2, low);
            vector.set(3, volume);
            vector.set(4, amount);
            List<Vector> vectors = allData.get(up);
            if (vectors == null) {
                //
                vectors = new ArrayList<>();
                allData.put(up, vectors);
            }
            vectors.add(vector);
        }
        return allData;
    }
    public static void main(String[] args) throws IOException {
        // 1, 加载数据
        HashMap<Integer, List<Vector>> allData = loadData();
        // 1.1 使用 70%的数据用作训练，剩下的30%用来测试
        HashMap<Integer, List<Vector>> trainDataMap = new HashMap<>();
        HashMap<Integer, List<Vector>> testDataMap = new HashMap<>();
        for (Map.Entry<Integer, List<Vector>> entry : allData.entrySet()) {
            int trainNum = (int) (0.7f * entry.getValue().size());
            trainDataMap.put(entry.getKey(), entry.getValue().subList(0, trainNum));
            testDataMap.put(entry.getKey(), entry.getValue().subList(trainNum, entry.getValue().size()));
        }
        // 2.1 构建贝叶斯的数据格式
        //  因为有三个品种所以是三个分类，同时特征向量有4个
        double[][] datas = new double[2][5];
        // 2.2 把同一个品种的同一列数据相加
        int i = 0;
        for (Map.Entry<Integer, List<Vector>> entry : trainDataMap.entrySet()) {
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
        DenseVector featureSum = new DenseVector(2);
        // 2.3 把同一个品种所有的特征值相加, 构建特征总值
        for (i = 0; i < 2; ++i) {
            double totalVal = 0;
            for (int j = 0; j < 5; ++j) {
                totalVal += datas[i][j];
            }
            featureSum.set(i, totalVal);
        }

        // 2.4 构建同一个特征值在所有品种的总和
        DenseVector labelSum = new DenseVector(5);
        for (i = 0; i < 5; ++i) {
            labelSum.set(i, datas[0][i] + datas[1][i]);
        }
        NaiveBayesModel naiveBayesModel = new NaiveBayesModel(weightMatrix, labelSum, featureSum, null, 1, true);
        StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(naiveBayesModel);
        // 3 测试数据
        int correctNum = 0;
        int totalTestNum = 0; //
        i = 0;
        for (Map.Entry<Integer, List<Vector>> entry : testDataMap.entrySet()) {
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
