package com.digquant;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import java.io.*;
import java.util.*;

public class PredictGenderSGD {
    private static final Multiset<String> male1 = ConcurrentHashMultiset.create();
    private static final Multiset<String> male2 = ConcurrentHashMultiset.create();
    //
    private static final Multiset<String> female1 = ConcurrentHashMultiset.create();
    private static final Multiset<String> female2 = ConcurrentHashMultiset.create();
    private static final ArrayList<Multiset<String>> males = new ArrayList<>();
    private static final ArrayList<Multiset<String>> females = new ArrayList<>();


    private static final HashSet<String> wordSet = new HashSet<>();


    private static final int FEATURES = 65535;

    private static final HashMap<String, Integer> nameMap = new HashMap<>();

    private static final HashMap<String, Integer> testMap = new HashMap<>();

    static class VectorHolder {
        Vector v;
        int target;
    }//

    public static void loadData() throws IOException {

        List<PredictGenderSGD1.VectorHolder> trainDataList = new ArrayList<>();//

        String path = "C:\\Users\\Administrator\\Desktop\\day7\\data\\name_data.txt";
        File file = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        // 把所有值都记录下来
        males.add(male1);
        males.add(male2);

        females.add(female1);
        females.add(female2);
        String line = reader.readLine();
        line = reader.readLine();

        while (line != null && line.length() > 0) {
            String[] lineArr = line.split(",");

            PredictGenderSGD1.VectorHolder vHolder = new PredictGenderSGD1.VectorHolder();//增加偏向值
            vHolder.v = new DenseVector(5);
            double y = Double.parseDouble(lineArr[2]);
            vHolder.v.set(1, y);
            trainDataList.add(vHolder);

            if (lineArr.length != 3) {
                continue;
            }
            String name = lineArr[1];

            int nameLen = name.length();

            for (int i = 0; i < nameLen; ++i) {
                // 分割数据集的
                if (nameMap.size() <= 91000) {
                    if ("1".equals(lineArr[2])) {
                        putNameInSet(name, males);
                        // 判断男女
                        nameMap.put(name, 1);
                    } else {
                        putNameInSet(name, females);
                        nameMap.put(name, 0);
                    }
                } else {
                    if ("1".equals(lineArr[2])) {
                        testMap.put(name, 1);
                    } else {
                        testMap.put(name, 0);
                    }
                }
            }
            line = reader.readLine();
        }
    }

    public static void main(String[] args) throws IOException {
        // 1, 加载数据
        loadData();
        // 配置学习算法
        OnlineLogisticRegression learningAlgorithm =  // 20 个分类
                new OnlineLogisticRegression(2, FEATURES, new L2()) // 判断是否已经是目标值
                        .alpha(1) // 学习率， 指数下降
                        .stepOffset(4000) // 衰减方式
                        .decayExponent(1.25) // 衰减率
                        .lambda(3.50e-5) // 正则化权重
                        .learningRate(48); // 初始学习率  //调整参数

        FeatureVectorEncoder encoder = new StaticWordValueEncoder("body"); // body什么的 只是给这个编码器起了名字而已，没有什么实际意义

        encoder.setProbes(1);

        FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");

        String testNamePath = "C:\\Users\\Administrator\\Desktop\\day7\\data\\name_test.txt";
        File file = new File(testNamePath);
        BufferedReader reader = new BufferedReader(new FileReader(file));

        String line = reader.readLine();
        line = reader.readLine();
        while (line != null && line.length() > 0) {
            String[] lineArr = line.split(",");
            if (lineArr.length != 2) {
                continue;
            }
            line = reader.readLine();
        }

        String testGenderPath = "C:\\Users\\Administrator\\Desktop\\day7\\data\\sample_submit.csv";
        file = new File(testGenderPath);
        reader = new BufferedReader(new FileReader(file));

        line = reader.readLine();
//        line = reader.readLine();
        while (line != null && line.length() > 0) {
            String[] lineArr = line.split(",");
            if (lineArr.length != 2) {
                continue;
            }
            line = reader.readLine();
        }

        for (int pass = 0; pass < 10; ++pass) {       //增加循环次数
            //
            for (Map.Entry<String, Integer> entry : nameMap.entrySet()) {
                String name = entry.getKey();
                int val = entry.getValue();
                Vector v = makeVector(encoder, bias, name);
                learningAlgorithm.train(val, v);
            }

        }

        ArrayList<String> names = new ArrayList<>();
        names.add("丽静");
        names.add("雅茵");
        names.add("晓静");
        names.add("春玲");
        names.add("晓意");
        names.add("惠君");
        names.add("嘉敏");
        names.add("慧");
        for (String name : names) {
            predictName(learningAlgorithm, encoder, bias, name, true);
        }

        int correct = 0;
        for (Map.Entry<String, Integer> entry : testMap.entrySet()) {
            Vector p = predictName(learningAlgorithm, encoder, bias, entry.getKey(), false);
            if (p.maxValueIndex() == entry.getValue()) {
                ++correct;
            }
        }
        System.out.printf("correct = %f \n", correct * 1.0 / testMap.size());
    }
    static Vector makeVector(FeatureVectorEncoder encoder, FeatureVectorEncoder bias, String name) {

        Vector v = new RandomAccessSparseVector(FEATURES);
        char[] chs = name.toCharArray();
        bias.addToVector("", chs.length, v); // 空
        for (int i = 0; i < chs.length; ++i) {
            String word = chs[i] + "";

            Multiset<String> male = males.get(i);

            Multiset<String> female = females.get(i);

            int total = male.size() + female.size();
            // 获取下标

            // （每个字在各个性别里面出现的总次数 + 1) / (总字数 + 各个性别里面字出现次数的总和)


            encoder.addToVector(word  + "_male", ((male.count(word) + 1) * 1.0) /(male.size() + wordSet.size()) * FEATURES, v);


            encoder.addToVector(word  + "_female", ((female.count(word) + 1) * 1.0) /(female.size() + wordSet.size()) * FEATURES, v);  //特征值计算公式
//            encoder.addToVector(word +  "_female", (female.count(word) * 1.0) / total * FEATURES, v);



//            encoder.addToVector(word  + "_male", (male.count(word) * 1.0) / total * FEATURES, v);
//
//            encoder.addToVector(word +  "_female", (female.count(word) * 1.0) / total * FEATURES, v);
        }

        return v;
    }


    static Vector predictName(
            OnlineLogisticRegression learningAlgorithm,
            FeatureVectorEncoder encoder,
            FeatureVectorEncoder bias,
            String name, boolean output) {

        Vector preV = makeVector(encoder, bias, name);
        Vector p = new DenseVector(2);
        learningAlgorithm.classifyFull(p, preV);
        int estimated = p.maxValueIndex();
        if (output) {
            System.out.printf("name = %s, male = %s \n", name, estimated == 1 ? "男" : "女");
        }
        return p;
    }

    private static void putNameInSet(String name, ArrayList<Multiset<String>> males) {
        char[] chs = name.toCharArray();
        for (int i = 0; i < chs.length; ++i) {
            String word = chs[i] + "";
            males.get(i).add(word);
            wordSet.add(word);
        }
    }
}
