---
layout: post
title: "Model inference in seldon kubernetes cluster"
author: "Karthik"
categories: journal
tags: [documentation,sample]

---







In this blog post, I’m sharing my journey of installing Seldon Core from scratch on my Mac. I started by setting up a local Kubernetes cluster and installing all the necessary dependencies, then tested everything with a scikit-learn Iris classification model. Although I followed the official guide, I ran into a few bumps along the way.

Initially, I tried deploying Seldon Core using the Ambassador gateway, but that approach kept giving me trouble. Instead, I switched to Istio as my ingress gateway, which ended up working much better. In this guide, I’ll walk you through the steps I took to install Seldon Core with Istio, along with some troubleshooting tips for the issues I encountered.

I hope this guide helps anyone looking to set up Seldon Core locally!





#### 1. Create a Kubernetes cluster

```
kind create cluster --name seldon --image kindest/node:v1.24.7
```

```
Creating cluster "seldon" ...
 ✓ Ensuring node image (kindest/node:v1.24.7) 🖼 
 ✓ Preparing nodes 📦  
 ✓ Writing configuration 📜 
 ✓ Starting control-plane 🕹️ 
 ✓ Installing CNI 🔌 
 ✓ Installing StorageClass 💾 
Set kubectl context to "kind-seldon"
You can now use your cluster with:

kubectl cluster-info --context kind-seldon

Thanks for using kind! 😊
```



```
kubectl cluster-info --context kind-seldon
```

```

Kubernetes control plane is running at https://127.0.0.1:57701
CoreDNS is running at https://127.0.0.1:57701/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

```



___



#### 2. Istio-system

It manages the control plane components (such as the Istio ingress gateway, Pilot, Mixer, etc.). It keeps all Istio-related resources organized and separated from your application namespaces.

```
kubectl get svc -n istio-system
No resources found in istio-system namespace.

```



Istio installation

```
curl -L https://istio.io/downloadIstio | sh -
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   101  100   101    0     0    696      0 --:--:-- --:--:-- --:--:--   701
100  5124  100  5124    0     0  14096      0 --:--:-- --:--:-- --:--:-- 14096

Downloading istio-1.25.0 from https://github.com/istio/istio/releases/download/1.25.0/istio-1.25.0-osx-arm64.tar.gz ...

Istio 1.25.0 download complete!

The Istio release archive has been downloaded to the istio-1.25.0 directory.

To configure the istioctl client tool for your workstation,
add the /Users/karthikrajedran/Documents/seldon-core/istio-1.25.0/bin directory to your environment path variable with:
	 export PATH="$PATH:/Users/karthikrajedran/Documents/seldon-core/istio-1.25.0/bin"

Begin the Istio pre-installation check by running:
	 istioctl x precheck 

Try Istio in ambient mode
	https://istio.io/latest/docs/ambient/getting-started/
Try Istio in sidecar mode
	https://istio.io/latest/docs/setup/getting-started/
Install guides for ambient mode
	https://istio.io/latest/docs/ambient/install/
Install guides for sidecar mode
	https://istio.io/latest/docs/setup/install/

Need more information? Visit https://istio.io/latest/docs/ 

```



After istio installation, update the istio folder's bin directory to the beginning of the list of places your system searches for commands.

```
cd istio-1.25.0

export PATH=$PWD/bin:$PATH

istioctl install --set profile=demo -y
```

```
        |\          
        | \         
        |  \        
        |   \       
      /||    \      
     / ||     \     
    /  ||      \    
   /   ||       \   
  /    ||        \  
 /     ||         \ 
/______||__________\
____________________
  \__       _____/  
     \_____/        


The Kubernetes version v1.24.7 is not supported by Istio 1.25.0. The minimum supported Kubernetes version is 1.28.
Proceeding with the installation, but you might experience problems. See https://istio.io/latest/docs/releases/supported-releases/ for a list of supported versions.

✔ Istio core installed ⛵️                                                                                   
✔ Istiod installed 🧠                                                                                       
✔ Egress gateways installed 🛫                                                                               
✔ Ingress gateways installed 🛬                                                                             
✔ Installation complete        

```



```
kubectl label namespace default istio-injection=enabled

namespace/default labeled
```





```
vi gateway.yaml

apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway # use istio default controller
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"


```

```
kubectl apply -f gateway.yaml

gateway.networking.istio.io/seldon-gateway created

```



```
kubectl create namespace seldon-system

namespace/seldon-system created
```



#### 3. Seldon-core installation using helm chart

Helm chart is used to simplify deployments, keep track of the deployed version and can be easily shared through repositories. 

```
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system
    
NAME: seldon-core
LAST DEPLOYED: Sun Mar  9 19:11:48 2025
NAMESPACE: seldon-system
STATUS: deployed
REVISION: 1
TEST SUITE: None
```



```
kubectl get pods -n seldon-system
NAME                                         READY   STATUS              RESTARTS   AGE
seldon-controller-manager-55654bbdf6-fbmmr   0/1     ContainerCreating   0          6s
```



```
kubectl get svc -n istio-system
NAME                   TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)                                                                      AGE
istio-egressgateway    ClusterIP      10.96.66.11     <none>        80/TCP,443/TCP                                                               4m51s
istio-ingressgateway   LoadBalancer   10.96.153.8     <pending>     15021:31533/TCP,80:31102/TCP,443:30459/TCP,31400:30124/TCP,15443:30528/TCP   4m51s
istiod                 ClusterIP      10.96.134.160   <none>        15010/TCP,15012/TCP,443/TCP,15014/TCP                                        5m33s


```



```
kubectl get gateway -n istio-system
NAME             AGE
seldon-gateway   2m52s
```



```
kubectl get pods -n seldon-system  
NAME                                         READY   STATUS    RESTARTS   AGE
seldon-controller-manager-55654bbdf6-fbmmr   1/1     Running   0          62s
```



```
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080
Handling connection for 8080
Handling connection for 8080
Handling connection for 8080
Handling connection for 8080
Handling connection for 8080
Handling connection for 8080
Handling connection for 8080
Handling connection for 8080
Handling connection for 8080

```

The seldon core installation is completed. 

I would recommend to use the configurations in an .yaml file rather than directly pasting it on the command line. Yaml file keeps the indentation intact, this will avoid indentation errors. 



#### 4. Sample Machine learning trained model inference

Seldon Core lets you run predictions on your trained models using a simple API. It works with many model types—including computer vision and Hugging Face models—and you can make predictions using either REST or gRPC calls.

```
kubectl apply -f - << END                                                  
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-model
  namespace: seldon
spec:
  name: iris
  predictors:
  - graph:
      implementation: SKLEARN_SERVER
      modelUri: gs://seldon-models/v1.19.0-dev/sklearn/iris
      name: classifier
    name: default
    replicas: 1
END

seldondeployment.machinelearning.seldon.io/iris-model created

```



```

curl -X POST http://localhost:8080/seldon/seldon/iris-model/api/v1.0/predictions \
    -H 'Content-Type: application/json' \
    -d '{ "data": { "ndarray": [[1,2,3,4]] } }'

```

 Inference Output

```    

{"data":{"names":["t:0","t:1","t:2"],"ndarray":[[0.0006985194531162853,0.003668039039435755,0.9956334415074478]]},"meta":{"requestPath":{"classifier":"seldonio/sklearnserver:1.17.1"}}}

```





In this blog post, I set up a workflow with Seldon Core to run machine learning model predictions at a large scale. Using Kubernetes gives us a big advantage—managing and scaling ML inference becomes much easier. With tools like HPA and KEDA, the system can automatically adjust to the amount of traffic, helping to control infrastructure costs.

Seldon Core also provides a full set of machine learning operations tools, including observability, explainers, and more. The only challenge is managing the Kubernetes infrastructure. If your organization is comfortable with Kubernetes and has a team to manage it, Seldon Core is a great choice for scalable ML inference.

In future posts, I’ll dive deeper into using Seldon Core for computer vision tasks and compare it with other platforms like Kubeflow.

