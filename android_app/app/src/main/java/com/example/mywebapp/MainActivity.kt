package com.example.mywebapp

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    // TODO: 在这里修改你的 IP 地址或 网址
    // 注意：如果是 HTTP 开头，必须在 AndroidManifest.xml 中配置 android:usesCleartextTraffic="true"
    private val targetUrl = "http://47.245.121.54:5173"

    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)

        // 配置 WebView 设置
        webView.settings.apply {
            javaScriptEnabled = true // 启用 JS，大部分网页需要
            domStorageEnabled = true // 启用 DOM 存储，防止部分网页加载空白
            useWideViewPort = true   // 将图片调整到适合 webview 的大小
            loadWithOverviewMode = true // 缩放至屏幕大小
        }

        // 设置 WebViewClient，确保链接在当前 APP 内打开，而不是跳转到 Chrome 浏览器
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                // 返回 false 表示由 WebView 自己处理该 URL
                return false
            }
        }

        // 设置 WebChromeClient (可选，用于处理网页标题、进度条等)
        webView.webChromeClient = WebChromeClient()

        // 加载网页
        webView.loadUrl(targetUrl)

        // 处理物理返回键逻辑 (Android 13+ 推荐写法)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    // 如果网页可以后退，则后退网页
                    webView.goBack()
                } else {
                    // 否则退出 APP
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })
    }
}