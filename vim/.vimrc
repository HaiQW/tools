set hlsearch
set incsearch
set number                                                                                         
"set expandtab                                                                                      
                                                                                                   
set filetype=c                                                                                     
set tabstop=4                                                                                      
set shiftwidth=4                                                                                   
set smartindent                                                                                    
set cindent                                                                                        
filetype plugin indent on                                                                          
set completeopt=longest,menu                                                                       
syntax on                                                                                          
                                                                                                   
                                                                                                   
let Tlist_Show_One_File=1                                                                          
let Tlist_Exit_OnlyWindow=1                                                                        
let Tlist_Use_Right_Window=1                                                                       
let Tlist_Auto_Open=1

set nocompatible
source $VIMRUNTIME/vimrc_example.vim
"source $VIMRUNTIME/mswin.vim
"behave mswin
set ruler
"set nu
syntax on
syntax enable
set ts=4
set noerrorbells
set sw=4
"set autoindent
"set smartindent
filetype on

set foldlevel=100

"set ignorecase smartcase
set so=5
colo desert
autocmd FileType c,py  setl fdm=syntax | setl fen
"let g:sh_fold_enabled= 1
set nobackup
"set cursorline
"set cursorcolumn
:runtime macros/matchit.vim
au  BufNewFile,BufRead  *.t2t   set ft=txt2tags
"set foldcolumn=4
if has('mouse')
    set mouse-=a 
    "set mouse=a 
endif
nmap # I#<Esc>j
map  <CTRL-n> :tabnew<CR>
set encoding=utf-8
set termencoding=utf-8
set fileencodings=ucs-bom,utf-8,gb18030,gbk,gb2312,big5,euc-jp,euc-kr,latin1,cp936

autocmd BufNewFile *.sh,*.pl,*.py exec ":call SetTitle()" 
"autocmd BufWrite   *.sh,*.pl,*.py exec ":call ModifyTitle()"
autocmd BufWrite *.sh,*pl,*py if getline(6) != "# Modify Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") || split(getline(7))[3] != strftime("%F") | call ModifyTitle() | endif

autocmd BufNewFile,BufRead *.py exec ":call SetTable()"
func SetTable()
    set expandtab
    set tabstop=4
    set shiftwidth=4
endfunc

func SetTitle()
    if &filetype == 'sh'
        call setline(1, "\#!/bin/sh") 
        call append(line("."), "\#****************************************************************#") 
        call append(line(".")+1, "\# ScriptName: ".expand("%") ) 
        call append(line(".")+2, "\# Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") )
        call append(line(".")+3, "\# Create Date: ".strftime("%F %R")) 
        call append(line(".")+4, "\# Modify Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") )
        call append(line(".")+5, "\# Modify Date: ".strftime("%F %R"))
        call append(line(".")+6, "\# Function: " ) 
        call append(line(".")+7, "\#***************************************************************#") 
        call append(line(".")+8, "") 
        :8
    elseif &filetype == 'perl'
        call setline(1, "\#!/usr/bin/perl") 
        call append(line("."), "\#****************************************************************#") 
        call append(line(".")+1, "\# ScriptName: ".expand("%") ) 
        call append(line(".")+2, "\# Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") ) 
        call append(line(".")+3, "\# Create Date: ".strftime("%F %R")) 
        call append(line(".")+4, "\# Modify Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") )
        call append(line(".")+5, "\# Modify Date: ".strftime("%F %R"))
        call append(line(".")+6, "\# Function: ") 
        call append(line(".")+7, "\#***************************************************************#") 
        call append(line(".")+8, "") 
        :8
    elseif &filetype == 'python'
        call setline(1, "\#!/usr/bin/python") 
        call append(line("."), "\#****************************************************************#") 
        call append(line(".")+1, "\# ScriptName: ".expand("%") ) 
        call append(line(".")+2, "\# Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") ) 
        call append(line(".")+3, "\# Create Date: ".strftime("%F %R")) 
        call append(line(".")+4, "\# Modify Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") )
        call append(line(".")+5, "\# Modify Date: ".strftime("%F %R"))
        call append(line(".")+6, "\# Function: ") 
        call append(line(".")+7, "\#***************************************************************#") 
        call append(line(".")+8, "") 
        :8
    endif
endfunc

func ModifyTitle()
    if getline(6) =~ "# Modify Author:.*"
        call setline(6, "\# Modify Author: ".expand("$SHTERM_REAL_USER@alibaba-inc.com") )
        call setline(7, "\# Modify Date: ".strftime("%F %R"))
    endif
endfunc
