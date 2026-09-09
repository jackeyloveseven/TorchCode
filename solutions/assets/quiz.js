/* TorchCode 学习工作区 · 测验组件 */

document.addEventListener('DOMContentLoaded', () => {

  /* -------- 多选题 -------- */
  document.querySelectorAll('.quiz-question').forEach(q => {
    const options  = q.querySelectorAll('.quiz-option');
    const feedback = q.querySelector('.quiz-feedback');
    const correct  = q.dataset.correct;

    options.forEach(opt => {
      opt.addEventListener('click', () => {
        if (opt.classList.contains('disabled')) return;

        const isCorrect = opt.dataset.key === correct;

        options.forEach(o => {
          o.classList.add('disabled');
          if (o.dataset.key === correct) o.classList.add('correct');
          else if (o === opt && !isCorrect) o.classList.add('incorrect');
        });

        if (feedback) {
          feedback.classList.add('show', isCorrect ? 'correct' : 'incorrect');
          feedback.textContent = isCorrect
            ? (q.dataset.feedbackCorrect  || '✓ 正确！')
            : (q.dataset.feedbackIncorrect || '✗ 不对，正确答案已高亮。');
        }
      });
    });
  });

  /* -------- 填空题 -------- */
  document.querySelectorAll('.fill-blank').forEach(ex => {
    const input    = ex.querySelector('.exercise-input');
    const btn      = ex.querySelector('.check-btn');
    const feedback = ex.querySelector('.quiz-feedback');
    const answers  = (ex.dataset.answers || '').split('|').map(s => s.trim().toLowerCase());

    const check = () => {
      const val = (input.value || '').trim().toLowerCase().replace(/\s+/g, ' ');
      const ok  = answers.includes(val);

      input.classList.toggle('correct', ok);
      input.classList.toggle('incorrect', !ok);

      if (feedback) {
        feedback.className = `quiz-feedback show ${ok ? 'correct' : 'incorrect'}`;
        feedback.textContent = ok
          ? (ex.dataset.feedbackCorrect || '✓ 正确！')
          : (ex.dataset.hint
              ? `✗ 再想想。提示：${ex.dataset.hint}`
              : '✗ 不对，再试试。');
      }
    };

    btn?.addEventListener('click', check);
    input?.addEventListener('keydown', e => { if (e.key === 'Enter') check(); });
  });

  /* -------- 解法展示 -------- */
  document.querySelectorAll('.solution-toggle-btn').forEach(btn => {
    const targetId = btn.dataset.target;
    const target   = document.getElementById(targetId);
    btn.addEventListener('click', () => {
      const open = target?.classList.toggle('show');
      btn.textContent = open ? '▲ 隐藏参考解法' : '▼ 查看参考解法（先自己想想！）';
    });
  });

});
