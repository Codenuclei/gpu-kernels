# 1) Hide markdown syntax (###, **) in hover popups
cat >> ~/.config/nvim/plugin/lsp-hover.lua <<'LUA'

vim.api.nvim_create_autocmd('FileType', {
  pattern = 'markdown',
  callback = function() vim.wo.conceallevel = 2 end,
})
LUA

# 2) Render Doxygen (\brief, \p) properly, clangd 21+ only
clangd --version
cd ~/gpu-kernels && cat >> .clangd <<'EOF'
Documentation:
  CommentFormat: Doxygen
EOF

# If ### still shows, check the markdown parsers are installed:
#   nvim -c 'checkhealth vim.treesitter'
