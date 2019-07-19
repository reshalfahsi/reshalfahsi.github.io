---
layout: post
title: Markdown Cheatsheet
---

This post is intended as a quick reference and showcase.   
For more complete reference, check out [John Gruber's original spec](http://daringfireball.net/projects/markdown/) and the [Github-flavored Markdown](http://github.github.com/github-flavored-markdown/) info page.

<!--break-->

### Table of Contents
   - [Headers](#header)  
   - [Emphasis](#emphasis)  
   - [Lists](#list)  
   - [Task Lists](#task)  
   - [Links](#link)  
   - [Images](#image)  
   - [Code and Syntax Highlighting](#code)  
   - [Tables](#table)  
   - [Blockquotes](#quote)  
   - [Inline HTML](#html)  
   - [Emojis](#emoji)  
   - [Horizontal Rules](#hr)  
   - [Youtube Videos](#youtube)  

<a name="header"/>

## Headers
```no-highlight
# First Header
## This is second header
### This is third header
###### smallest header

Alternative first header
========================
Alternative second header
-------------------------
```

# First Header
## This is second header
### This is third header
###### smallest header

Alternative first header
========================
Alternative second header
-------------------------



<a name="emphasis"/>

## Emphasis
```no-highlight
*italic* _another italic_  
**bold** __another bold__  
**_bold italic_**  
~~strikethrough~~  
```

*italic* _another italic_  
**bold** __another bold__  
**_bold italic_**  
~~strikethrough~~  

<a name="list"/>

## Lists
```no-highlight
1. first ordered list
2. second item
4. third item
   * unordered sub-list
     - second nested list
1. new list

   creating properly indented paragraph within list item.  
   notice the blank line above below list and the leading space.  
   at least one space needed, but we'll use three here to also align in raw markdown.  

* using asterisks
- or minus
+ or plus
```

1. first ordered list
2. second item
4. third item
   * unordered sub-list
     - second nested list

1. new list

   creating properly indented paragraph within list item.  
   notice the blank line above below list and the leading space.  
   at least one space needed, but we'll use three here to also align in raw markdown.  

* using asterisks
- or minus
+ or plus

<a name="task"/>

## Task Lists

```no-highlight
- [x] Finish changes
- [ ] Push commits
- [ ] Edit readme
- [ ] \(Optional) With escape
```

- [x] Finish changes
- [ ] Push commits
- [ ] Edit readme
- [ ] \(Optional) With escape

<a name="link"/>

## Links
two ways of creating links
```no-highlight
[inline-style link](https://google.com)  
[inline-style link with title](https://google.com "Google's Homepage")  
[reference link][Arbitary case-insensitive]  
[relative reference to a repo file](../blob/master/LICENSE) -- not supported in page  
[numbered reference][1]  
[link to itself]  

[arbitary case-insensitive]: https://facebook.com
[1]: http://github.com
[link to itself]: http://twitter.com
```

[inline-style link](https://google.com)  
[inline-style link with title](https://google.com "Google's Homepage")  
[reference link][Arbitary case-insensitive]  
[relative reference to a repo file](./README.md) -- not supported in page  
[numbered reference][1]  
[link to itself]  

[arbitary case-insensitive]: https://facebook.com
[1]: http://github.com
[link to itself]: http://twitter.com

<a name="image"/>

## Images
```no-highlight
Some images example  

Inline-style:  
![alt text](/images/404.jpg "404 Not Found")

Reference-style:  
![alt text][logo]

[logo]: https://triwahyuu.github.io/images/jekyll-logo.png "Jekyll's Logo"
```

Here's Jekyll's logo  

Inline-style:  
![alt text](/images/404.jpg "404 Not Found")

Reference-style:  
![alt text][logo]

[logo]: https://triwahyuu.github.io/images/jekyll-logo.png "Jekyll's Logo"

<a name="code"/>

## Code and Syntax Highlighting
Creating code and syntax highlighting inside markdown  

```no-highlight
Inline `code` using `back-ticks around` it  
```

Inline `code` using `back-ticks around` it  

Block of code using three back-ticks or indented with four spaces. back-ticks are recomended as they support syntax highlighting.  

<pre lang="no-highlight"><code>```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print s
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```
</code></pre>


```javascript
var s = "JavaScript syntax highlighting";
alert(s);
let a = 10;
```

```python
s = "Python syntax highlighting"
print s
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

To find the valid highlighting keyword see the [highlight.js demo page](https://highlightjs.org/static/demo/)

<a name="table"/>

## Tables
Tables aren't supported in Markdown spec.  
But they are part of GFM and Markdown Here supports them.  
Colons can be used to align columns.  

```no-highlight
| Tables        | Are              | Cool    |
| ------------- |:----------------:| -------:|
| col 3 is      | right-aligned    |   $1600 |
| col 2 is      | centered         |     $12 |
| zebra stripes | are neat         |      $1 |

There must be at least 3 dashes separating each header cell.  
The outer pipes (|) are optional, and you don't need to make the raw Markdown line up prettily. You can also use inline Markdown.  

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3
```

| Tables        | Are              | Cool    |
| ------------- |:----------------:| -------:|
| col 3 is      | right-aligned    |   $1600 |
| col 2 is      | centered         |     $12 |
| zebra stripes | are neat         |      $1 |

There must be at least 3 dashes separating each header cell.  
The outer pipes (|) are optional, and you don't need to make the raw Markdown line up prettily. You can also use inline Markdown.  

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

<a name="quote"/>

## Blockquotes
```no-highlight
> Blockquotes to emulate reply.
> This line is part of the same quotes.

Quote break.  

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote.
```

> Blockquotes to emulate reply.
> This line is part of the same quotes.

Quote break.  

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote.

<a name="html"/>

## Inline HTML
Using raw HTML in markdown.  
```html
<dl>
  <dt>Definition list</dt>
  <dd>Is something people use sometimes.</dd>

  <dt>Markdown in HTML</dt>
  <dd>Does *not* work **very** well. Use HTML <em>tags</em>.</dd>
</dl>
```

<dl>
  <dt>Definition list</dt>
  <dd>Is something people use sometimes.</dd>

  <dt>Markdown in HTML</dt>
  <dd>Does *not* work **very** well. Use HTML <em>tags</em>.</dd>
</dl>

You can also create an ASCII symbol using HTML Unicode (UTF-8).

```html
Pound &pound; Yen &yen;  

Using the hexadecimal or decimal value  

&#x24;Dollar&#36;  
&#x0023;Hashtag&#35;  
&#34;Quote&#x0022;&quot;  
```

Pound &pound; Yen &yen;  

Using the hexadecimal or decimal value  

&#x24;Dollar&#36;  
&#x0023;Hashtag&#35;  
&#34;Quote&#x0022;&quot;  

For more ASCII code reference check out [w3school HTML UTF-8](https://www.w3schools.com/charsets/ref_html_utf8.asp) and the more interractive version from [Toptal's HTML Arrow](https://www.toptal.com/designers/htmlarrows/).

<a name="emoji"/>

## Emojis
Creating emojis by typing `:EMOJICODE:`

```no-highlight
:+1: This PR looks great - it's ready to merge! :relieved:
```
:+1: This PR looks great - it's ready to merge! :relieved:

For full list of available emoji and the codes, check out [emoji-cheat-sheet.com](http://emoji-cheat-sheet.com)

<a name="hr"/>

## Horizontal Rule
```no-highlight
Three or more...

---
Hyphens

***
Asterisks

___
Underscores
```

Three or more...

---
Hyphens

***
Asterisks

___
Underscores

<a name="youtube"/>

## Youtube Videos
They can't be added directly but we can add an image with a link to the video  
```no-highlight
<a href="http://www.youtube.com/watch?feature=player_embedded&v=UbIPKktQGfU" target="_blank"><img src="/images/demo_vid.png" alt="IMAGE ALT TEXT HERE" width="480" height="360"/></a>
```
<a href="https://www.youtube.com/watch?v=P9KTu0k6qgQ" target="_blank"><img src="/images/demo_vid.png" alt="IMAGE ALT TEXT HERE" width="480" height="360"/></a>

or in pure markdown, but losing the image sizing and border  
```no-highlight
[![IMAGE ALT TEXT HERE](/images/demo_vid.png)](https://www.youtube.com/watch?v=UbIPKktQGfU)
```
[![IMAGE ALT TEXT HERE](/images/demo_vid.png)](https://www.youtube.com/watch?v=P9KTu0k6qgQ)

You can create an embedded youtube videos screenshoot from this [w3shools tutorial](https://www.w3schools.com/html/html_youtube.asp).
