---
layout: post
title: Raspbian Kernel Name Editing
---

<div class="message">
  Hello, World!~. こんにちは!! Have you ever heard about Raspberry Pi? Did you know what Operating System is? Have you ever heard about Linux or Raspbian? Did you know that you can change your Linux kernel name and version? Today I will explain you about these things, so keep reading this article and check it out!
</div>

<!--break-->

### Table of Contents
   - [Introduction](#introduction)  
   - [Raspberry Pi](#raspi)  
   - [Operating System](#os)  
   - [Linux for Raspberry Pi?](#linux)  
   - [Kernel Name and Version](#kernel)  
   - [Toolchain Requirements](#toolchain)  
   - [Kernel Name and Version Editing](#kerneledit)  
   - [Build Kernel](#build)  
   - [Check SDCard](#sdcard)  
   - [Finish](#finish)  

<a name="introduction"/>

## Introduction

This article main focus is explaining about Raspberry Pi, Operating System, Linux for Raspberry Pi a.k.a. Raspbian, and change the Linux kernel name and version. Another topic such as How to Install Operating System on Raspberry Pi and etc. will not be covered by this article. If you wonder about these topics you can click the following [link](https://google.com). However, this is not a very technical and complete article but still I hope you can understand and enjoy it!

<a name="raspi"/>

## Raspberry Pi

Back to my Senior Highschool day at SMA 3 Yogyakarta, I always wonder about something technology. Like what Steve Jobs say "Stay Foolish, Stay Hungry", I spent my time to search about cool stuff like Python, C++, Tensorflow, Arduino, Brilliant.org, 9GAG.com, 1CAK.com aaand many more. Before it was famous, I know Tensorflow at my last year of Senior Highschool and I didn't understand at all at that time. It was complicated. Beside Tensorflow, I also knew about Arduino. Even I have been created a project with my friend, Bagas, using Arduino to make a homemade Roomba. Then I join many groups and like many pages about Arduino and technology in Facebook. That's how I knew Raspberry Pi. At first, I think Arduino and Raspberry Pi are same. I kept this thinking until I enroll for college. I study at Electrical Engineering of Gadjah Mada University. I met many cool people there. One of them named Harit.

<a name="os"/>

## Operating System

Cum sociis natoque penatibus et magnis dis `code element` montes, nascetur ridiculus mus.

{% highlight js %}
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
// > 8
{% endhighlight %}

Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa.

### Lists

Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus.

* Praesent commodo cursus magna, vel scelerisque nisl consectetur et.
* Donec id elit non mi porta gravida at eget metus.
* Nulla vitae elit libero, a pharetra augue.

Donec ullamcorper nulla non metus auctor fringilla. Nulla vitae elit libero, a pharetra augue.

1. Vestibulum id ligula porta felis euismod semper.
2. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.
3. Maecenas sed diam eget risus varius blandit sit amet non magna.

Cras mattis consectetur purus sit amet fermentum. Sed posuere consectetur est at lobortis.

<dl>
  <dt>HyperText Markup Language (HTML)</dt>
  <dd>The language used to describe and define the content of a Web page</dd>

  <dt>Cascading Style Sheets (CSS)</dt>
  <dd>Used to describe the appearance of Web content</dd>

  <dt>JavaScript (JS)</dt>
  <dd>The programming language used to build advanced Web sites and applications</dd>
</dl>

Integer posuere erat a ante venenatis dapibus posuere velit aliquet. Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Nullam quis risus eget urna mollis ornare vel eu leo.

### Images

Quisque consequat sapien eget quam rhoncus, sit amet laoreet diam tempus. Aliquam aliquam metus erat, a pulvinar turpis suscipit at.

![placeholder](http://placehold.it/800x400 "Large example image")
![placeholder](http://placehold.it/400x200 "Medium example image")
![placeholder](http://placehold.it/200x200 "Small example image")

### Tables

Aenean lacinia bibendum nulla sed consectetur. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Upvotes</th>
      <th>Downvotes</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Totals</td>
      <td>21</td>
      <td>23</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Alice</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Charlie</td>
      <td>7</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

Nullam id dolor id nibh ultricies vehicula ut id elit. Sed posuere consectetur est at lobortis. Nullam quis risus eget urna mollis ornare vel eu leo.

-----

