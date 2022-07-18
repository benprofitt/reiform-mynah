export default function getDate(t: number): string {
  const d = new Date(t*1000)
  const day = d.getDate()
  const months = [ 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
  const month = months[d.getMonth()]
  const year = d.getFullYear()
  return `${month} ${day} ${year}`
}